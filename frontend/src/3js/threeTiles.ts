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
  status: 'OK' | 'ESTIMATED' | 'CONFLICT' | 'UNCHECKED' | 'INSUFFICIENT'
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

/** Relative agreement required between a record and its room (float64 rounding only). */
const SAME = 1e-6

const agrees = (a: number, b: number) =>
  Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <= SAME * Math.max(1, Math.abs(a), Math.abs(b))

/**
 * Why a surface record may NOT be drawn: it must be the room object's own
 * values -- the same camera (focal length, principal point), the same single
 * millimetre scale (floor: its plane sits at the room's camera height; walls:
 * the room frame, already in mm) -- and one tile size in mm. The 3D view never
 * fixes a mismatch with a scale of its own; it refuses to draw instead.
 * Empty list = consistent.
 */
export function roomConsistency(data: ThreeJobData, surface: string): string[] {
  const record = data.surfaces[surface]
  const room = data.room
  if (!record) return [`no record for ${surface}`]
  if (!room?.camera?.K) return ['this render carries no room geometry to check it against']

  const problems: string[] = []
  const K = room.camera.K
  if (!agrees(record.camera.focal_px, K[0][0])) problems.push(`focal ${record.camera.focal_px} != room ${K[0][0]}`)
  if (!agrees(record.camera.cx, K[0][2]) || !agrees(record.camera.cy, K[1][2]))
    problems.push('principal point differs from the room camera')

  const mmPerUnit = record.grid.mm_per_unit
  if (!agrees(record.meters_per_unit * 1000, mmPerUnit)) problems.push('two scales inside one record')
  if (record.kind === 'floor' && !agrees(record.plane.d_units * mmPerUnit, room.camera_height_mm))
    problems.push(`floor at ${record.plane.d_units * mmPerUnit} mm, room camera height ${room.camera_height_mm} mm`)
  if (record.kind === 'wall' && !agrees(mmPerUnit, 1)) problems.push(`wall on its own scale (${mmPerUnit} mm/unit)`)

  if (!(record.tile.width_mm > 0) || !(record.tile.height_mm > 0)) problems.push('no tile size in mm')
  return problems
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
//   the pixel's camera ray -> the recorded plane -> (u, v) -> rotate + offset
//   -> x mm_per_unit = grid mm -> grid fractions -> flip -> sample tile
//   (bilinear, wrap, texture-space v squash) -> grout at a constant on-screen
//   width, faded below it -> room lighting -> opacity. Then the strict clip:
//   only white mask pixels.
//
// The grid is computed per fragment from the ray, as the engine computes it per
// pixel, not interpolated across the quad: a wall reaching its vanishing line
// has a quad kilometres deep, and float32 interpolation across it shifted the
// tile pattern (measured: 23% of one wall's pixels > 8 levels off).
//
// For the same reason the surface is rasterized as a quad covering the whole
// image, not as its kilometres-deep plane quad (whose far edge float32 missed
// by a pixel): every pixel gets a fragment, the mask decides which keep a tile,
// and the recorded camera + plane decide what each one shows.

const vertexShader = /* glsl */ `
void main() {
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`

/** Two triangles over the whole image, in clip space. */
function coverageGeometry(): THREE.BufferGeometry {
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0]), 3))
  geometry.setIndex([0, 1, 2, 0, 2, 3])
  return geometry
}

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
uniform float focal;
uniform vec2 principal;
uniform vec3 planeN;
uniform float planeD;
uniform vec3 eU;
uniform vec3 eV;
uniform vec2 rotCosSin;
uniform vec2 offsetUnits;
uniform float mmPerUnit;
out vec4 fragColor;

void main() {
  // The image pixel under this fragment (row 0 at the top, like the masks).
  ivec2 p = ivec2(int(floor(gl_FragCoord.x)), int(imageSize.y) - 1 - int(floor(gl_FragCoord.y)));

  // Strict clip: tiles exist only on this surface's white mask pixels.
  if (texelFetch(maskTex, p, 0).r < 0.5) discard;

  // The engine evaluates pixel i at image coordinate i: that pixel's ray meets
  // the plane n.P + d = 0 at P; u = P.e_u, v = P.e_v (core.raycast / three_layer).
  vec3 ray = vec3((vec2(p) - principal) / focal, 1.0);
  vec3 P = ray * (-planeD / dot(planeN, ray));
  float u = dot(P, eU);
  float v = dot(P, eV);
  // cos/sin come from JS (float64): GLSL's built-in trig precision is
  // implementation-defined, and ~1e-4 of it shifts a far tile visibly.
  float c = rotCosSin.x;
  float s = rotCosSin.y;
  vec2 grid = (vec2(u * c - v * s, u * s + v * c) + offsetUnits) * mmPerUnit;

  // Grid fractions (core.uv.to_grid_fractions).
  vec2 fr = fract(grid / tileMm);
  fr = mix(fr, 1.0 - fr, flip);

  // OpenCV remap, bilinear with wrap, v squashed by the engine's stretch.
  vec2 t = vec2(fr.x * tilePx.x, (fr.y / stretch) * tilePx.y);
  vec4 tile = texture(tileTex, (t + 0.5) / tilePx);
  vec3 col = tile.rgb * 255.0;

  // Grout (core.composite.apply_grout): local mm per pixel from derivatives.
  if (groutOn > 0.5) {
    float mmU = max(length(vec2(dFdx(grid.x), dFdy(grid.x))), 1e-6);
    float mmV = max(length(vec2(dFdx(grid.y), dFdy(grid.y))), 1e-6);
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

const vecLength = (v: number[]) => Math.hypot(...v) || 1
const unitNormal = (v: number[]) => v.map((x) => x / vecLength(v)) as [number, number, number]

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
      focal: { value: record.camera.focal_px },
      principal: { value: new THREE.Vector2(record.camera.cx, record.camera.cy) },
      planeN: { value: new THREE.Vector3(...unitNormal(record.plane.normal)) },
      planeD: { value: record.plane.d_units / vecLength(record.plane.normal) },
      eU: { value: new THREE.Vector3(...record.plane.e_u) },
      eV: { value: new THREE.Vector3(...record.plane.e_v) },
      rotCosSin: { value: new THREE.Vector2(Math.cos(record.grid.rotation_rad), Math.sin(record.grid.rotation_rad)) },
      offsetUnits: { value: new THREE.Vector2(...record.grid.offset_units) },
      mmPerUnit: { value: record.grid.mm_per_unit },
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
      const mesh = new THREE.Mesh(coverageGeometry(), surfaceMaterial(record, item))
      mesh.frustumCulled = false
      scene.add(mesh)

      const camera = new THREE.PerspectiveCamera()
      const c = record.camera
      // The shader casts each pixel's ray with the engine's own intrinsics (the
      // `focal` / `principal` uniforms, pixel i at coordinate i). This camera is
      // the same projection, kept for the scene; the coverage quad ignores it.
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
