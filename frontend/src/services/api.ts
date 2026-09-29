/**
 * Backend integration layer.
 *
 * The Python pipeline in ../scripts/ is not exposed over HTTP yet, so nothing
 * here is wired up to a live server. This file defines the contract the
 * frontend expects, so connecting the backend later means pointing
 * VITE_API_BASE_URL at the server that implements POST /generate.
 *
 * Nothing in this module fabricates a result. When no backend is configured,
 * isBackendConfigured() returns false and the UI says so explicitly.
 */

import type {
  ComposeResponse,
  FlowResponse,
  GenerateRequest,
  GenerateResponse,
  SegmentsResponse,
  SurfacesResponse,
} from '../types'

/** Set VITE_API_BASE_URL in frontend/.env.local once the backend exists. */
export const API_BASE_URL: string = import.meta.env.VITE_API_BASE_URL ?? ''

export const GENERATE_ENDPOINT = '/generate'

export const FLOW_ENDPOINT = '/pipeline/flow'

export const SEGMENTS_ENDPOINT = '/pipeline/segments'

export const SEGMENT_ENDPOINT = '/segment'

export const SEGMENT_START_ENDPOINT = '/segment/start'
export const SEGMENT_STATUS_ENDPOINT = '/segment/status'

export function isBackendConfigured(): boolean {
  return API_BASE_URL.trim().length > 0
}

export class ApiError extends Error {
  readonly status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

/**
 * Builds the multipart body for POST /generate.
 *
 * Exported on its own so the exact field names stay verifiable without a
 * running server:
 *   room_image, tile_image, room_width, room_length, room_height,
 *   tile_width, tile_height, rotation, surface
 */
export function buildGenerateFormData(request: GenerateRequest): FormData {
  const form = new FormData()

  form.append('room_image', request.room_image)
  form.append('tile_image', request.tile_image)
  form.append('room_width', String(request.room_width))
  form.append('room_length', String(request.room_length))
  form.append('room_height', String(request.room_height))
  form.append('tile_width', String(request.tile_width))
  form.append('tile_height', String(request.tile_height))
  form.append('rotation', String(request.rotation))
  form.append('surface', request.surface)

  // Only sent when the caller asks for a grout width, so the backend's own
  // default stays in charge otherwise.
  if (request.grout_mm !== undefined) {
    form.append('grout_mm', String(request.grout_mm))
  }

  // Sent only when a subset is chosen; empty means every wall.
  if (request.walls?.length) {
    form.append('walls', request.walls.join(','))
  }

  // Sent only once the room has been cleaned. The backend verifies the photo's
  // bytes against the job before reusing anything, so a stale id is harmless.
  if (request.job_id) {
    form.append('job_id', request.job_id)
  }

  // One wall on its own. Sent only when asked for.
  if (request.wall_id) {
    form.append('wall_id', request.wall_id)
  }

  return form
}

/**
 * Pulls the server's own explanation out of a failed response.
 *
 * The backend rejects requests it cannot honestly render (an uncalibrated room,
 * for instance) with a FastAPI `{ detail }` body. Showing that text beats
 * showing a bare status code, so the reason reaches the user unchanged.
 */
async function readErrorDetail(response: Response): Promise<string> {
  const fallback = `Generation failed (${response.status} ${response.statusText})`

  try {
    const body = (await response.json()) as { detail?: unknown }

    if (typeof body.detail === 'string' && body.detail.trim()) return body.detail

    return fallback
  } catch {
    return fallback
  }
}

/**
 * Calls POST /generate.
 *
 * Throws if no backend is configured — the caller must check
 * isBackendConfigured() first and show the UI's not-connected state.
 */
export async function generateVisualization(
  request: GenerateRequest,
  signal?: AbortSignal,
): Promise<GenerateResponse> {
  if (!isBackendConfigured()) {
    throw new ApiError(
      'No backend configured. Set VITE_API_BASE_URL to the server that implements POST /generate.',
      0,
    )
  }

  const response = await fetch(`${API_BASE_URL}${GENERATE_ENDPOINT}`, {
    method: 'POST',
    body: buildGenerateFormData(request),
    signal,
  })

  if (!response.ok) {
    throw new ApiError(await readErrorDetail(response), response.status)
  }

  return (await response.json()) as GenerateResponse
}

/** Shared GET helper for the read-only pipeline endpoints. */
async function getJson<T>(endpoint: string, signal?: AbortSignal): Promise<T> {
  if (!isBackendConfigured()) {
    throw new ApiError('No backend configured.', 0)
  }

  const response = await fetch(`${API_BASE_URL}${endpoint}`, { signal })

  if (!response.ok) {
    throw new ApiError(await readErrorDetail(response), response.status)
  }

  return (await response.json()) as T
}

/** The pipeline's nine stages, with a preview image for each one that ran. */
export function fetchPipelineFlow(signal?: AbortSignal): Promise<FlowResponse> {
  return getJson<FlowResponse>(FLOW_ENDPOINT, signal)
}

/**
 * Everything the pipeline segmented out of the room: the props as transparent
 * PNGs, plus the wall/floor/ceiling split.
 */
export function fetchSegments(signal?: AbortSignal): Promise<SegmentsResponse> {
  return getJson<SegmentsResponse>(SEGMENTS_ENDPOINT, signal)
}

/**
 * Extracts an uploaded room photo into two transparent PNGs: `ALL_OBJECTS.png`
 * (every object, mirrors included) and `MIRRORS_ONLY.png` (only the mirrors).
 * Both are the size of the photo, with objects left at their original
 * coordinates, and they are the only object images the request produces.
 *
 * Works on any photo — unlike POST /generate it needs no calibrated camera.
 */
/** How often a running segmentation is checked on. */
const SEGMENT_POLL_MS = 2000

/**
 * Segment a room without holding one long HTTP request open.
 *
 * `/segment` takes 90-500 seconds. A reverse proxy will not wait that long —
 * Cloudflare answers 524 at about 100 seconds, and because its error page
 * carries no CORS header the browser reports the whole thing as
 * "Failed to fetch". Measured against the public tunnel: HTTP 524 after 125.6s
 * for work the backend finished correctly in 94.69s.
 *
 * So the work is started, and progress is polled. Every request here lasts
 * milliseconds, which no proxy timeout can reach. The resolved value is the
 * same object `segmentRoom` returns, so callers are unaffected.
 */
export async function segmentRoomAsync(
  file: File,
  signal?: AbortSignal,
  onProgress?: (elapsedSeconds: number) => void,
): Promise<SegmentsResponse> {
  if (!isBackendConfigured()) {
    throw new ApiError('No backend configured.', 0)
  }

  const form = new FormData()
  form.append('room_image', file)

  const started = await fetch(`${API_BASE_URL}${SEGMENT_START_ENDPOINT}`, {
    method: 'POST',
    body: form,
    signal,
  })

  if (!started.ok) {
    throw new ApiError(await readErrorDetail(started), started.status)
  }

  const { job_id: token } = (await started.json()) as { job_id: string }

  // No overall deadline: the caller's AbortSignal is the way out, and a slow
  // room is slow rather than broken.
  for (;;) {
    if (signal?.aborted) {
      throw new DOMException('Aborted', 'AbortError')
    }

    await new Promise((resolve) => setTimeout(resolve, SEGMENT_POLL_MS))

    const response = await fetch(
      `${API_BASE_URL}${SEGMENT_STATUS_ENDPOINT}/${token}`,
      { signal },
    )

    if (!response.ok) {
      throw new ApiError(await readErrorDetail(response), response.status)
    }

    const payload = await response.json()

    if (payload.status === 'running') {
      onProgress?.(payload.elapsed_s ?? 0)
      continue
    }

    return payload as SegmentsResponse
  }
}


export async function segmentRoom(
  file: File,
  signal?: AbortSignal,
): Promise<SegmentsResponse> {
  if (!isBackendConfigured()) {
    throw new ApiError('No backend configured.', 0)
  }

  const form = new FormData()
  form.append('room_image', file)

  const response = await fetch(`${API_BASE_URL}${SEGMENT_ENDPOINT}`, {
    method: 'POST',
    body: form,
    signal,
  })

  if (!response.ok) {
    throw new ApiError(await readErrorDetail(response), response.status)
  }

  return (await response.json()) as SegmentsResponse
}


export const SURFACES_ENDPOINT = '/surfaces'
export const COMPOSE_ENDPOINT = '/compose'

/** POST a form and read JSON, turning a failure into an ApiError with the server's reason. */
async function postForm<T>(endpoint: string, form: FormData, signal?: AbortSignal): Promise<T> {
  if (!isBackendConfigured()) {
    throw new ApiError('No backend configured.', 0)
  }

  const response = await fetch(`${API_BASE_URL}${endpoint}`, { method: 'POST', body: form, signal })

  if (!response.ok) {
    throw new ApiError(await readErrorDetail(response), response.status)
  }

  return (await response.json()) as T
}

/**
 * The floor and every wall a cleaned room offers, each with a select dot.
 * `jobId` is the room's Clean Room run; `roomFile` the same photograph.
 */
export function fetchSurfaces(
  roomFile: File,
  jobId: string,
  signal?: AbortSignal,
): Promise<SurfacesResponse> {
  const form = new FormData()
  form.append('room_image', roomFile)
  form.append('job_id', jobId)
  return postForm<SurfacesResponse>(SURFACES_ENDPOINT, form, signal)
}

/**
 * The photograph with only the selected surfaces tiled. `floorJob` and
 * `wallJob` are /generate runs for surface=floor and surface=wall.
 */
export function composeSurfaces(
  request: { floorJob?: string; wallJob?: string; floor: boolean; walls: string[] },
  signal?: AbortSignal,
): Promise<ComposeResponse> {
  const form = new FormData()
  if (request.floorJob) form.append('floor_job', request.floorJob)
  if (request.wallJob) form.append('wall_job', request.wallJob)
  form.append('floor', String(request.floor))
  form.append('walls', request.walls.join(','))
  return postForm<ComposeResponse>(COMPOSE_ENDPOINT, form, signal)
}

/**
 * The photograph with each listed surface's tiles, each from its own render:
 * `layers` pairs a /generate job with the surface to take from it.
 */
export function composeLayers(
  layers: { job: string; surface: string }[],
  signal?: AbortSignal,
): Promise<ComposeResponse> {
  const form = new FormData()
  form.append('layers', layers.map((item) => `${item.job}:${item.surface}`).join(','))
  return postForm<ComposeResponse>(COMPOSE_ENDPOINT, form, signal)
}
