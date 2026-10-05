/** Shared types for the AI Tile Visualizer frontend. */

export type Surface = 'floor' | 'wall' | 'both'

export type Rotation = 0 | 45 | 90 | 135

export const ROTATIONS: Rotation[] = [0, 45, 90, 135]

export const SURFACES: { value: Surface; label: string }[] = [
  { value: 'floor', label: 'Floor' },
  { value: 'wall', label: 'Wall' },
  { value: 'both', label: 'Both' },
]

/** Tile dimensions in millimetres, matching the backend tile spec. */
export interface TileSize {
  width: number
  height: number
}

export interface TileSizePreset extends TileSize {
  id: string
  label: string
}

export const TILE_PRESETS: TileSizePreset[] = [
  { id: '600x1200', label: '600 × 1200 mm', width: 600, height: 1200 },
  { id: '800x1600', label: '800 × 1600 mm', width: 800, height: 1600 },
  { id: '1200x2400', label: '1200 × 2400 mm', width: 1200, height: 2400 },
  { id: '2400x4800', label: '2400 × 4800 mm', width: 2400, height: 4800 },
]

export const CUSTOM_TILE_ID = 'custom'

/** An uploaded file plus the object URL used for its preview. */
export interface UploadedImage {
  file: File
  previewUrl: string
}

/** Room dimensions in feet. Kept as strings so the inputs can be empty. */
export interface RoomDimensions {
  width: string
  length: string
  height: string
}

export interface VisualizerForm {
  roomImage: UploadedImage | null
  tileImage: UploadedImage | null
  room: RoomDimensions
  tilePresetId: string
  customTile: { width: string; height: string }
  rotation: Rotation
  surface: Surface
}

/** Field-keyed validation messages. Empty object means the form is valid. */
export type ValidationErrors = Partial<
  Record<
    | 'roomImage'
    | 'tileImage'
    | 'roomWidth'
    | 'roomLength'
    | 'roomHeight'
    | 'tileSize'
    | 'rotation'
    | 'surface',
    string
  >
>

/** How the room size is known: typed by the user, or estimated from the photo. */
export type RoomMode = 'auto' | 'manual'

/**
 * The payload the backend will receive at POST /generate.
 * Field names match the agreed multipart form keys exactly.
 */
export interface GenerateRequest {
  room_image: File
  tile_image: File
  /**
   * Room size in feet. Only the dimensions the user typed are sent: an empty
   * one is left for the backend to estimate from the photo.
   */
  room_width?: number
  room_length?: number
  room_height?: number
  /**
   * MANUAL when any room dimension was typed, AUTO when none was. Optional:
   * without it the backend infers the mode from which dimensions arrived.
   */
  room_mode?: RoomMode
  tile_width: number
  tile_height: number
  rotation: Rotation
  surface: Surface
  /**
   * Grout width in millimetres. Optional: the backend keeps its existing 5 mm
   * default when this is absent, so an older caller is unaffected.
   */
  grout_mm?: number
  /**
   * The job a previous POST /segment produced for this exact photograph.
   *
   * With it, the backend reuses that run's accepted object masks, clean room
   * and surfaces instead of segmenting the photo a second time — so what is
   * restored over the tiles is cut along the same boundaries as the
   * ALL_OBJECTS.png on screen, and the render takes seconds rather than a
   * minute and a half. Omitted or stale, the backend runs the full pipeline.
   */
  job_id?: string
  /**
   * One wall on its own, as POST /surfaces lists it ("wall-2"). With
   * surface 'wall' and a Clean Room job_id, only that wall is rendered.
   */
  wall_id?: string
  /**
   * Which walls to tile. Omitted or empty means every wall the room has,
   * which is the behaviour this endpoint has always had.
   */
  walls?: WallLabel[]
}

/** The walls `live_scene` builds for an uploaded room. */
export type WallLabel = 'left' | 'right' | 'back'

export const WALL_LABELS: { value: WallLabel; label: string }[] = [
  { value: 'left', label: 'Left Wall' },
  { value: 'back', label: 'Back Wall' },
  { value: 'right', label: 'Right Wall' },
]

/** What the backend returns from POST /generate. Nothing is faked locally. */
export interface GenerateResponse {
  original_image_url: string
  result_image_url: string
  job_id?: string
  /**
   * How the camera was obtained. `fitted` is the pipeline's measured camera;
   * `estimated` is inferred from the uploaded photo's segmented floor.
   */
  geometry?: { method: 'fitted' | 'estimated'; [key: string]: unknown }
  /** What the render actually did, including per-wall pixel coverage. */
  stats?: {
    wall_pixels_by_plane?: Record<string, number>
    floor_pixels?: number
    [key: string]: unknown
  }
  /** Server-side caveats worth showing the user verbatim. */
  notes?: string[]
  /**
   * The surfaces this render tiled, each with its select dot. Null when the
   * tile engine did not render the request (no Clean Room run).
   */
  surfaces?: SurfaceInfo[] | null
}

/**
 * One selectable surface: the floor, or one wall as the tile engine splits
 * WALL_MASK. `id` is stable for a room: "floor", or "wall-<engine index>".
 */
export interface SurfaceInfo {
  id: string
  kind: 'floor' | 'wall'
  /** "Floor", or "Wall 1", "Wall 2"… numbered left to right. */
  label: string
  /** Where the select dot sits, in pixels of the rendered image. */
  dot: [number, number]
  pixels: number
}

/** POST /surfaces: what a cleaned room offers, before any tile is chosen. */
export interface SurfacesResponse {
  job_id: string
  /** [width, height] of the image the dots are measured on. */
  canvas: [number, number]
  surfaces: SurfaceInfo[]
}

/** POST /compose: the photograph with only the selected surfaces tiled. */
export interface ComposeResponse {
  result_image_url: string
  original_image_url: string
  selection: { floor: boolean; walls: string[] }
  tiled_pixels: number
  /** Always 0: nothing outside the selected surfaces differs from the photo. */
  outside_selection_pixels: number
}

/** One pipeline stage as reported by GET /pipeline/flow. */
export interface FlowStage {
  step: number
  id: string
  title: string
  purpose: string
  /** `live` means this server recomputes the stage per request. */
  status: 'done' | 'live' | 'unavailable'
  preview_url: string | null
}

export interface FlowResponse {
  stages: FlowStage[]
}

/**
 * One transparent PNG the backend produced — an object layer, or a surface.
 *
 * For the two object layers (`all_objects`, `mirrors`) the bbox is the whole
 * room image: they are full-canvas layers holding every object at its original
 * coordinates, so they composite straight back over a retiled room.
 */
export interface Segment {
  id: string
  name: string
  source_stage: string
  pixels: number
  /** [x0, y0, x1, y1] in master-input pixels. */
  bbox: [number, number, number, number]
  cutout_url?: string
  mask_url?: string
  /** For a surface: the fraction of the frame it covers. */
  coverage?: number
  /** For a layer: the file it is written as, e.g. `ALL_OBJECTS.png`. */
  filename?: string
  /** For a layer: the objects merged into it. */
  members?: string[]
}

/** One instance the segmentation found, and what became of it. Debug only. */
export interface Detection {
  name: string
  status: 'object' | 'mirror' | 'excluded' | 'skipped' | 'failed'
  pixels?: number
  bbox?: [number, number, number, number]
  score?: number
  layers?: string[]
  reason?: string
}

export interface SegmentsResponse {
  room_url: string
  /** Present when the photo is a room-data room: where the result came from. */
  room_data?: { room_id: string; source: string; message: string | null; status?: string }
  /** Exactly two entries: the all-objects layer and the mirrors layer. */
  objects: Segment[]
  surfaces: Segment[]
  object_count: number
  /** Every detection considered, accepted or not, with rejection reasons. */
  detections?: Detection[]
  /** `live` means the uploaded photo was segmented just now. */
  source?: 'live' | 'pipeline'
  job_id?: string
  success?: boolean
  all_objects_png?: string
  mirrors_only_png?: string
  /**
   * The room with its objects removed and the surfaces behind them rebuilt —
   * an opaque image, not a transparency. Null when the inpainting stage could
   * not run, in which case `clean_room.reason` says why.
   */
  clean_room_png?: string | null
  clean_room?: {
    available?: boolean
    reason?: string
    model?: string
    /** Pixels reconstructed: the object masks, grown by `grow_px`. */
    filled_px?: number
    grow_px?: number
    windows?: number
    checks?: {
      /** Pixels altered outside the object masks. Always 0. */
      outside_changed?: number
      dark_px?: number
      surface_delta?: Record<string, number>
    }
  }
  counts?: {
    raw_detections: number
    accepted: number
    mirrors: number
    objects: number
    rejected: number
  }
  /**
   * The floor/wall split of the clean room: two binary masks, white where the
   * surface is, at the source image's own resolution. Null when the stage
   * could not run, in which case `floor_wall.reason` says why.
   */
  floor_mask_png?: string | null
  wall_mask_png?: string | null
  floor_wall?: {
    available?: boolean
    reason?: string
    /** Which image the masks were measured on. */
    source?: string
    on_clean_room?: boolean
    floor_mask_url?: string
    wall_mask_url?: string
    floor_overlay_url?: string
    wall_overlay_url?: string
    stats?: {
      backend?: string
      /** Always `['floor', 'wall']` — the only two classes this stage detects. */
      classes?: string[]
      /** [height, width] of the image the masks belong to. */
      source_shape?: [number, number]
      floor_pixels?: number
      wall_pixels?: number
      floor_coverage?: number
      wall_coverage?: number
      /** Always 0: the two masks are mutually exclusive. */
      overlap_pixels?: number
    }
  }
  timings?: Record<string, number>
  metadata_url?: string
  debug?: { enabled: boolean; files: string[] }
}

export type StepStatus = 'pending' | 'active' | 'done'

export interface PipelineStep {
  id: string
  label: string
  status: StepStatus
}

/** Frontend progress stages, shown while the backend works. */
export const PIPELINE_STEPS: { id: string; label: string }[] = [
  { id: 'upload_room', label: 'Uploading room' },
  { id: 'upload_tile', label: 'Uploading tile' },
  { id: 'process_room', label: 'Processing room' },
  { id: 'apply_tile', label: 'Applying tile' },
  { id: 'restore_objects', label: 'Restoring objects' },
]

export type GenerationStatus = 'idle' | 'running' | 'done' | 'error' | 'not_connected'

// ---------------------------------------------------------------------------
// Showroom UI — room picker and tile catalogue.
// ---------------------------------------------------------------------------

/** One demo room offered on the landing screen. */
export interface DemoRoom {
  id: string
  name: string
  category: string
  src: string
  thumb: string
}

/** One product in the tile catalogue rail. */
export interface CatalogueTile {
  id: string
  sku: string
  name: string
  size: string
  finish: string
  badge: string | null
  /** Physical tile size in mm — what POST /generate is given. */
  width: number
  height: number
  src: string
  thumb: string
}

export interface Catalogue {
  rooms: DemoRoom[]
  tiles: CatalogueTile[]
}

/** Grout presets, in millimetres. The backend renders grout at a fixed 5 mm. */
export const GROUT_OPTIONS: { value: number; label: string }[] = [
  { value: 2, label: '2 mm' },
  { value: 5, label: '5 mm' },
  { value: 8, label: '8 mm' },
]

/** Which surface the studio is currently painting. */
export type StudioSurface = Exclude<Surface, 'both'> | 'both'
