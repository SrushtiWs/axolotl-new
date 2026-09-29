/**
 * Selectable surface indicators, drawn on the room itself.
 *
 * One control per surface the room offers: the floor, and every wall the tile
 * engine found — however many that is. Selecting one puts the current tile on
 * that surface straight away and makes it follow the next tile chosen;
 * deselecting it keeps the tiles it has and stops changing them. Every surface
 * is independent, so the floor and each wall can carry different tiles.
 *
 * The floor is a labelled pill ("Floor"); each wall is a dot. Dark with a
 * tick = selected, white = not. A wall's name is on its dot as the accessible
 * name and the tooltip rather than printed beside it, so a row of walls does
 * not cover the tiles they are showing.
 *
 * Each dot sits where the backend measured that surface to be — the point
 * deepest inside it — so it is always on its own surface.
 */

export interface SurfaceMarker {
  /** "floor", or "wall-<engine index>". */
  id: string
  label: string
  /** Percentage position within the image. */
  x: number
  y: number
  selected: boolean
  /** Its tiles are being rendered right now. */
  pending?: boolean
  /** "pill" for the floor, "dot" for a wall. */
  shape?: 'pill' | 'dot'
  /** Why this surface could not be tiled, when it could not. */
  failed?: string
}

/** Where the image actually sits inside the stage, in pixels. */
export interface ImageFrame {
  left: number
  top: number
  width: number
  height: number
}

interface Props {
  markers: SurfaceMarker[]
  onToggle: (marker: SurfaceMarker) => void
  /**
   * The displayed image's box within the stage. The image is letterboxed, so
   * a dot placed by percentage of the whole stage would miss its surface.
   */
  frame?: ImageFrame | null
}

export function SurfaceMarkers({ markers, onToggle, frame }: Props) {
  if (markers.length === 0) return null

  return (
    <div
      className="markers"
      role="group"
      aria-label="Surfaces to tile"
      style={
        frame
          ? { inset: 'auto', left: frame.left, top: frame.top, width: frame.width, height: frame.height }
          : undefined
      }
    >
      {markers.map((marker) => {
        const tip = marker.pending
          ? `${marker.label} — laying tiles…`
          : marker.failed
            ? `${marker.label} could not be tiled: ${marker.failed}. Tap to try again.`
            : marker.selected
              ? `${marker.label} — selected. New tiles go here; tap to keep its tiles and stop changing it.`
              : `Select the ${marker.label.toLowerCase()}`

        const className = [
          'marker',
          marker.shape === 'pill' && 'pill',
          marker.selected && 'on',
          marker.pending && 'pending',
          marker.failed && !marker.selected && 'failed',
        ]
          .filter(Boolean)
          .join(' ')

        return (
          <button
            key={marker.id}
            type="button"
            role="switch"
            aria-checked={marker.selected}
            aria-busy={marker.pending || undefined}
            aria-label={marker.label}
            className={className}
            style={{ left: `${marker.x}%`, top: `${marker.y}%` }}
            onClick={() => onToggle(marker)}
            title={tip}
          >
            {marker.shape === 'pill' && <span className="marker-label">{marker.label}</span>}
            <span className="marker-dot" aria-hidden="true">
              {marker.selected && (
                <svg viewBox="0 0 12 12" aria-hidden="true">
                  <path
                    d="M2.5 6.3 4.8 8.6 9.5 3.9"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.8"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              )}
            </span>
          </button>
        )
      })}
    </div>
  )
}
