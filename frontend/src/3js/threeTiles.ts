/**
 * Optional 3D tile layer.
 *
 * Draws the tiles of already-rendered surfaces with Three.js, from the render
 * state the tile engine recorded next to each render (`three.json`). Nothing
 * here detects, estimates or guesses: the camera, the planes, the grid
 * (rotation, anchor, metric scale), the texture squash, the grout and the
 * lighting are the 2D engine's own values, replayed on the GPU.
 *
 * Scene units: 1 unit = 1 metre, so a 600 × 1200 mm tile is 0.6 × 1.2.
 *
 * Frames. The engine's camera frame is x right, y down, z forward, principal
 * point at (cx, cy). Three.js looks down -z with y up, so a point (x, y, z)
 * becomes (x, -y, -z). The projection matrix is built from the engine's focal
 * length and principal point directly, so a 3D point lands on exactly the
 * image pixel the engine projected it to.
 */

import * as THREE from 'three'

// ------------------------------------------------------------------ the data

export interface ThreeSurfaceRecord {
  kind: 'floor' | 'wall'
  camera: { focal_px: number; cx: number; cy: number; image_size: [number, number] }
  plane: { normal: number[]; d_units: number; e_u: number[]; e_v: number[] }
  meters_per_unit: number
  quad_units: number[][]
  quad_grid_mm: number[][]
  grid: { rotation_rad: number; offset_units: number[]; mm_per_unit: number }
  tile: {
    width_mm: number
    height_mm: number
    size_px: [number, number]
    flip: [boolean, boolean]
    height_stretch: number
  }
  grout: {
    width_px: [number, number]
    ink: [number, number]
    max_half_fraction: number
    color_rgb: [number, number, number]
    width_mm: number
  } | null
  lighting: { blend: number; average_brightness: number; opacity: number }
  mask: string
}

/**
 * The canonical room a render used (tiles_backend/perspective_engine/room/geometry.py):
 * one metric scale (the camera height), the camera, and the room box in mm. Every
 * surface record below was rendered on this same scale, so the 3D view needs no
 * scale of its own.
 */
export interface RoomGeometry {
  status: 'OK' | 'ESTIMATED' | 'CONFLICT' | 'INSUFFICIENT'
  width_mm: number | null
  length_mm: number | null
  height_mm: number | null
  width_source: string
  length_source: string
  height_source: string
  camera_height_mm: number
  camera_height_source: string
  camera: { K: number[][]; R_room_to_camera: number[][]; position_mm?: number[]; pitch_deg: number; yaw_deg: number; roll_deg: number }
  corners_3d: Record<string, number[]>
  projected_corners_2d: Record<string, number[] | null>
  reprojection_error_mean_px: number | null
  confidence: number
}

export interface ThreeJobData {
  version: number
  /** The room geometry this render used; absent for renders made before it existed. */
  room?: RoomGeometry | null
  image_size: [number, number]
  photo: string
  base: string
  tile: string
  objects: string
  surfaces: Record<string, ThreeSurfaceRecord>
}

/** One surface to draw: which render it came from and that render's record. */
export interface ThreeLayer {
  /** URL of the job folder, ending in "/". */
  jobBase: string
  surface: string
  data: ThreeJobData
}

const jobCache = new Map<string, Promise<ThreeJobData>>()

/**
 * `three.json` of one render job (cached: a job's render never changes).
 * Deliberately not abortable: the cached request is shared by every caller,
 * so one caller cancelling must not fail it for the others. Callers drop a
 * result they no longer want.
 */
export function loadThreeJob(url: string): Promise<ThreeJobData> {
  let pending = jobCache.get(url)
  if (!pending) {
    pending = fetch(url).then((response) => {
      if (!response.ok) throw new Error(`3D data unavailable (${response.status})`)
      return response.json() as Promise<ThreeJobData>
    })
    pending.catch(() => jobCache.delete(url))
    jobCache.set(url, pending)
  }
  return pending
}

// ------------------------------------------------------------------ geometry

/** Engine camera frame (units) -> Three.js world (metres). */
function toWorld(point: number[], metersPerUnit: number): THREE.Vector3 {
  return new THREE.Vector3(point[0] * metersPerUnit, -point[1] * metersPerUnit, -point[2] * metersPerUnit)
}

/**
 * Pinhole projection from the engine's intrinsics. Checked by construction:
 * a camera-frame point (X, Y, Z) maps to pixel (cx + f X / Z, cy + f Y / Z).
 */
export function projectionFromIntrinsics(
  f: number, cx: number, cy: number, width: number, height: number, near = 0.01, far = 100000,
): THREE.Matrix4 {
  const m = new THREE.Matrix4()
  m.set(
    (2 * f) / width, 0, 1 - (2 * cx) / width, 0,
    0, (2 * f) / height, (2 * cy) / height - 1, 0,
    0, 0, -(far + near) / (far - near), (-2 * far * near) / (far - near),
    0, 0, -1, 0,
  )
  return m
}

/** The surface's plane as a quad in metres, carrying the tile grid (mm) at each corner. */
export function surfaceGeometry(record: ThreeSurfaceRecord): THREE.BufferGeometry {
  const corners = record.quad_units.map((p) => toWorld(p, record.meters_per_unit))
  const positions = new Float32Array(corners.flatMap((v) => [v.x, v.y, v.z]))
  const grid = new Float32Array(record.quad_grid_mm.flat())

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  geometry.setAttribute('gridMm', new THREE.BufferAttribute(grid, 2))
  geometry.setIndex([0, 1, 2, 0, 2, 3])
  return geometry
}

// ------------------------------------------------------------------ shading
//
// Per pixel, the same steps as the 2D engine (core/uv.py, core/composite.py):
//   grid fractions -> flip -> sample tile (bilinear, wrap, texture-space v
//   squash) -> grout at a constant on-screen width, faded below it ->
//   room lighting -> opacity. Then the strict clip: only white mask pixels.

const vertexShader = /* glsl */ `
in vec2 gridMm;
out vec2 vGrid;
void main() {
  vGrid = gridMm;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`

const fragmentShader = /* glsl */ `
precision highp float;
uniform sampler2D tileTex;
uniform sampler2D maskTex;
uniform sampler2D baseTex;
uniform vec2 imageSize;
uniform vec2 tileMm;
uniform vec2 tilePx;
uniform float stretch;
uniform vec2 flip;
uniform float groutOn;
uniform vec2 groutWidthPx;
uniform vec2 groutInk;
uniform float groutMaxHalf;
uniform vec3 groutColor;
uniform float lightBlend;
uniform float avgBright;
uniform float opacity;
in vec2 vGrid;
out vec4 fragColor;

void main() {
  // The image pixel under this fragment (row 0 at the top, like the masks).
  ivec2 p = ivec2(int(floor(gl_FragCoord.x)), int(imageSize.y) - 1 - int(floor(gl_FragCoord.y)));

  // Strict clip: tiles exist only on this surface's white mask pixels.
  if (texelFetch(maskTex, p, 0).r < 0.5) discard;

  // Grid fractions (core.uv.to_grid_fractions).
  vec2 fr = fract(vGrid / tileMm);
  fr = mix(fr, 1.0 - fr, flip);

  // OpenCV remap, bilinear with wrap, v squashed by the engine's stretch.
  vec2 t = vec2(fr.x * tilePx.x, (fr.y / stretch) * tilePx.y);
  vec4 tile = texture(tileTex, (t + 0.5) / tilePx);
  vec3 col = tile.rgb * 255.0;

  // Grout (core.composite.apply_grout): local mm per pixel from derivatives.
  if (groutOn > 0.5) {
    float mmU = max(length(vec2(dFdx(vGrid.x), dFdy(vGrid.x))), 1e-6);
    float mmV = max(length(vec2(dFdx(vGrid.y), dFdy(vGrid.y))), 1e-6);
    float distU = min(fr.x, 1.0 - fr.x) * tileMm.x;
    float distV = min(fr.y, 1.0 - fr.y) * tileMm.y;
    float halfU = min(0.5 * groutWidthPx.x * mmU, groutMaxHalf * tileMm.x);
    float halfV = min(0.5 * groutWidthPx.y * mmV, groutMaxHalf * tileMm.y);
    float fu = groutInk.x * (1.0 - clamp((distU - halfU) / mmU, 0.0, 1.0));
    float fv = groutInk.y * (1.0 - clamp((distV - halfV) / mmV, 0.0, 1.0));
    float g = max(fu, fv);
    col = col * (1.0 - g) + groutColor * g;
  }

  // Room lighting (core.composite.composite), from the render's base image.
  vec3 room = texelFetch(baseTex, p, 0).rgb * 255.0;
  float bright = (room.r + room.g + room.b) / 3.0;
  float light = clamp(bright / (avgBright + 0.1), 0.15, 2.2);
  col = col * (1.0 - lightBlend) + col * light * lightBlend;

  float alpha = tile.a * opacity;
  vec3 outCol = clamp(col, 0.0, 255.0) / 255.0;
  fragColor = vec4(outCol * alpha, alpha);   // premultiplied, for the transparent canvas
}
`

function exactTexture(texture: THREE.Texture, filter: THREE.MagnificationTextureFilter, wrap: THREE.Wrapping) {
  texture.flipY = false // row 0 = top, as in the engine's arrays
  texture.colorSpace = THREE.NoColorSpace // raw values, no sRGB conversion
  texture.generateMipmaps = false // OpenCV samples the full-resolution tile
  texture.magFilter = filter
  texture.minFilter = filter
  texture.wrapS = wrap
  texture.wrapT = wrap
  texture.needsUpdate = true
  return texture
}

export function surfaceMaterial(
  record: ThreeSurfaceRecord,
  textures: { tile: THREE.Texture; mask: THREE.Texture; base: THREE.Texture },
): THREE.ShaderMaterial {
  const grout = record.grout
  return new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    vertexShader,
    fragmentShader,
    side: THREE.DoubleSide,
    depthTest: false,
    depthWrite: false,
    transparent: false,
    blending: THREE.NoBlending,
    uniforms: {
      tileTex: { value: textures.tile },
      maskTex: { value: textures.mask },
      baseTex: { value: textures.base },
      imageSize: { value: new THREE.Vector2(...record.camera.image_size) },
      tileMm: { value: new THREE.Vector2(record.tile.width_mm, record.tile.height_mm) },
      tilePx: { value: new THREE.Vector2(...record.tile.size_px) },
      stretch: { value: record.tile.height_stretch || 1 },
      flip: { value: new THREE.Vector2(record.tile.flip[0] ? 1 : 0, record.tile.flip[1] ? 1 : 0) },
      groutOn: { value: grout ? 1 : 0 },
      groutWidthPx: { value: new THREE.Vector2(...(grout?.width_px ?? [0, 0])) },
      groutInk: { value: new THREE.Vector2(...(grout?.ink ?? [0, 0])) },
      groutMaxHalf: { value: grout?.max_half_fraction ?? 0.4 },
      groutColor: { value: new THREE.Vector3(...(grout?.color_rgb ?? [0, 0, 0])) },
      lightBlend: { value: record.lighting.blend },
      avgBright: { value: record.lighting.average_brightness },
      opacity: { value: record.lighting.opacity },
    },
  })
}

// ------------------------------------------------------------------ renderer

/**
 * Renders a set of surface layers onto one canvas at the photo's own
 * resolution. Each surface is drawn with its own render's camera (floor and
 * wall renders may carry different intrinsics), so every surface lines up
 * with the photograph exactly where its 2D render did.
 */
export class TileLayerRenderer {
  private renderer: THREE.WebGLRenderer
  private loader = new THREE.TextureLoader()
  private textures = new Map<string, Promise<THREE.Texture>>()
  private disposed = false

  constructor(canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      alpha: true,
      premultipliedAlpha: true,
      antialias: false, // one sample per pixel, as the engine
      preserveDrawingBuffer: true, // lets the result be read back and checked
    })
    this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace // no conversion on output
    this.renderer.setPixelRatio(1)
    this.renderer.setClearColor(0x000000, 0)
    this.loader.setCrossOrigin('anonymous')
  }

  private texture(url: string, filter: THREE.MagnificationTextureFilter, wrap: THREE.Wrapping) {
    const key = `${url}|${filter}|${wrap}`
    let pending = this.textures.get(key)
    if (!pending) {
      pending = this.loader.loadAsync(url).then((t) => exactTexture(t, filter, wrap))
      this.textures.set(key, pending)
    }
    return pending
  }

  /** Draw exactly these layers (anything drawn before is cleared). */
  async render(layers: ThreeLayer[]): Promise<void> {
    if (!layers.length) {
      this.renderer.clear()
      return
    }
    const [width, height] = layers[0].data.image_size
    this.renderer.setSize(width, height, false)

    const prepared = await Promise.all(
      layers.map(async (layer) => {
        const record = layer.data.surfaces[layer.surface]
        if (!record) return null
        const [tile, mask, base] = await Promise.all([
          this.texture(layer.jobBase + layer.data.tile, THREE.LinearFilter, THREE.RepeatWrapping),
          this.texture(layer.jobBase + record.mask, THREE.NearestFilter, THREE.ClampToEdgeWrapping),
          this.texture(layer.jobBase + layer.data.base, THREE.NearestFilter, THREE.ClampToEdgeWrapping),
        ])
        return { record, tile, mask, base }
      }),
    )
    if (this.disposed) return

    this.renderer.autoClear = false
    this.renderer.clear()

    for (const item of prepared) {
      if (!item) continue
      const { record } = item
      const scene = new THREE.Scene()
      const mesh = new THREE.Mesh(surfaceGeometry(record), surfaceMaterial(record, item))
      mesh.frustumCulled = false
      scene.add(mesh)

      const camera = new THREE.PerspectiveCamera()
      const c = record.camera
      // The engine evaluates pixel i at image coordinate i; a WebGL fragment's
      // centre is at i + 0.5. Shifting the principal point by half a pixel makes
      // each fragment sample the exact point the engine sampled for that pixel.
      camera.projectionMatrix.copy(
        projectionFromIntrinsics(c.focal_px, c.cx + 0.5, c.cy + 0.5, c.image_size[0], c.image_size[1]),
      )
      camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert()

      this.renderer.render(scene, camera)

      mesh.geometry.dispose()
      ;(mesh.material as THREE.Material).dispose()
    }
  }

  dispose() {
    this.disposed = true
    for (const pending of this.textures.values()) pending.then((t) => t.dispose()).catch(() => undefined)
    this.textures.clear()
    this.renderer.dispose()
  }
}
