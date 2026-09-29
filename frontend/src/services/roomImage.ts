/**
 * Is this photograph a room?
 *
 * Two stages, and the split between them is the point.
 *
 * **Here, instantly** — only the checks that are certain: the file is an
 * image, it is big enough to segment, and its shape is not a strip. A room
 * photo is roughly 4:3, 3:2 or 16:9; a tile slab is 2:1 or taller. This costs
 * nothing and catches the obvious mistake before any upload happens.
 *
 * **In the backend, authoritatively** — how much floor the segmentation finds.
 * That is the signal that actually separates the two cases, and it was
 * measured rather than assumed. Across fifteen rooms and fourteen tiles:
 *
 *     rooms   floor coverage 0.079 - 0.368   (lowest 7.9%)
 *     tiles   floor coverage 0.000 - 0.039   (highest 3.9%)
 *
 * A clean gap, with no overlap. A pixel-statistics classifier was tried here
 * first and thrown away: marble veining reads as "structure" at thumbnail
 * scale, so it rejected only three of the fourteen tiles while risking real
 * rooms. Guessing badly is worse than not guessing, so this file no longer
 * guesses — `floorVerdict` below judges the number the backend reports.
 */

export interface RoomImageVerdict {
  ok: boolean
  reason?: string
}

/** Widest a room photo is expected to be, in either orientation. */
const MAX_ASPECT = 2.6

/** Smallest short edge worth segmenting. */
const MIN_EDGE = 240

/**
 * Floor coverage below which a photo is not a room.
 *
 * Set in the measured gap between the two populations — comfortably above the
 * busiest tile (3.9%) and below the emptiest room (7.9%).
 */
export const MIN_FLOOR_COVERAGE = 0.05

export const INVALID_ROOM =
  'Invalid room image. Please upload a proper room/interior image.'

function load(file: File): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file)
    const image = new Image()

    image.onload = () => {
      URL.revokeObjectURL(url)
      resolve(image)
    }

    image.onerror = () => {
      URL.revokeObjectURL(url)
      reject(new Error('That file could not be read as an image.'))
    }

    image.src = url
  })
}

/** The instant checks. Passing this does not mean the photo is a room. */
export async function validateRoomImage(file: File): Promise<RoomImageVerdict> {
  if (!file.type.startsWith('image/')) {
    return { ok: false, reason: `${INVALID_ROOM} (that file is not an image)` }
  }

  let image: HTMLImageElement

  try {
    image = await load(file)
  } catch (cause) {
    return { ok: false, reason: cause instanceof Error ? cause.message : INVALID_ROOM }
  }

  const { naturalWidth: w, naturalHeight: h } = image

  if (Math.min(w, h) < MIN_EDGE) {
    return {
      ok: false,
      reason: `${INVALID_ROOM} (this image is ${w}×${h}; a room photo needs at least ${MIN_EDGE}px on its short side)`,
    }
  }

  const ratio = Math.max(w, h) / Math.min(w, h)

  if (ratio > MAX_ASPECT) {
    return {
      ok: false,
      reason: `${INVALID_ROOM} (${w}×${h} is a ${ratio.toFixed(1)}:1 strip — that is a tile or panel shape, not a room)`,
    }
  }

  return { ok: true }
}

/**
 * The authoritative check, on the floor coverage the backend reports.
 *
 * `coverage` is the fraction of the frame the segmentation called floor.
 */
export function floorVerdict(coverage: number | undefined): RoomImageVerdict {
  if (coverage === undefined) {
    return { ok: false, reason: `${INVALID_ROOM} (no floor was found in this photo)` }
  }

  if (coverage < MIN_FLOOR_COVERAGE) {
    return {
      ok: false,
      reason: `${INVALID_ROOM} (only ${(coverage * 100).toFixed(1)}% of it reads as floor — a room needs a visible stretch of ground to tile)`,
    }
  }

  return { ok: true }
}
