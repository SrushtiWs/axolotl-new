/**
 * Clearing the room: the original photo beside what was lifted out of it.
 *
 * This is a view onto POST /segment, which already produces exactly two
 * transparent layers — `ALL_OBJECTS.png` and `MIRRORS_ONLY.png`, both at the
 * photo's own size with every object left at its original coordinates. Nothing
 * about that pipeline is touched here; this only calls it and shows what comes
 * back.
 *
 * One honest limitation is surfaced rather than hidden. The backend does not
 * inpaint: it never reconstructs the wall behind a sofa, so there is no
 * "emptied room" photograph to show. What *is* produced is the separation —
 * the objects lifted off the room — which is what the tile renderer composites
 * back over the new floor. The panel says so instead of implying a repaint.
 */

import { useEffect, useRef, useState } from 'react'
import { CleanRoom } from './CleanRoom'
import { segmentRoomAsync } from '../services/api'
import { floorVerdict } from '../services/roomImage'
import type { SegmentsResponse, UploadedImage } from '../types'

interface Props {
  room: UploadedImage
  /**
   * Told what the segmentation found, so the studio can gate its surface
   * controls on it. `null` means "not cleared yet"; a verdict of invalid means
   * the photograph has no floor in it and is not a room.
   */
  onCleared?: (result: SegmentsResponse | null, invalid: string | null) => void
  /** Told whenever a clearing run starts or stops, so the parent can block
   *  its own inference while this one is using the CPU. */
  onBusy?: (busy: boolean) => void
  /** Set while the parent is rendering. Both are minute-long CPU jobs and
   *  running them together makes each several times slower. */
  blocked?: boolean
  /** Start cleaning as soon as a room opens, without waiting for the button. */
  autoStart?: boolean
}

export function RoomAnalysis({ room, onCleared, onBusy, blocked, autoStart }: Props) {
  const [data, setData] = useState<SegmentsResponse | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  /** Seconds the running segmentation has been going, for the button. */
  const [elapsed, setElapsed] = useState(0)

  const abort = useRef<AbortController | null>(null)

  // A new room invalidates whatever the last one produced.
  useEffect(() => {
    setData(null)
    setError(null)
    abort.current?.abort()
    onCleared?.(null, null)

    // Start cleaning as soon as the room opens, so its floor and walls are
    // ready to select without anyone having to press the button first.
    if (autoStart) void clearRoom(true)
    // `onCleared` is intentionally not a dependency: it is a notification, and
    // re-running this on every parent render would clear the panel constantly.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [room.previewUrl])

  useEffect(() => () => abort.current?.abort(), [])

  useEffect(() => {
    onBusy?.(busy)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [busy])

  /** `useSaved`: opening the room may use its saved room-data result; Re-run never does. */
  async function clearRoom(useSaved = false) {
    abort.current?.abort()

    const controller = new AbortController()
    abort.current = controller

    setBusy(true)
    setError(null)
    setElapsed(0)

    try {
      const response = await segmentRoomAsync(room.file, controller.signal, setElapsed, useSaved)

      setData(response)

      // The authoritative room check: how much floor the segmentation found.
      const floor = response.surfaces?.find((item) => item.id === 'floor')

      const verdict = floorVerdict(floor?.coverage)

      if (!verdict.ok) setError(verdict.reason ?? null)
      else if (response.room_data?.message) setError(response.room_data.message)

      onCleared?.(response, verdict.ok ? null : (verdict.reason ?? 'Not a room.'))
    } catch (cause) {
      // Superseded by a newer click: that run owns the button now.
      if (abort.current !== controller) return

      setError(
        cause instanceof Error && cause.name === 'AbortError'
          ? 'Cleaning was interrupted before it finished. Press Clean Room to try again.'
          : cause instanceof Error
            ? cause.message
            : 'Could not clear this room.',
      )
    } finally {
      // Release the button unless a newer request has taken over.
      //
      // This used to test `controller.signal.aborted`, which left `busy` true
      // forever whenever a run was aborted by anything other than a fresh
      // click — a page reload, a dev-server hot reload, a room change. The
      // button then sat disabled reading "Cleaning…" and the panel looked
      // broken with no way back. Comparing against the live controller is the
      // question actually being asked: has someone else started a newer run?
      if (abort.current === controller) setBusy(false)
    }
  }

  const objects = data?.objects.find((layer) => layer.id === 'all_objects')
  const mirrors = data?.objects.find((layer) => layer.id === 'mirrors')

  return (
    <section className="analysis">
      <header className="analysis-head">
        <h3>Clean Room</h3>

        <button
          type="button"
          className="btn tiny"
          onClick={() => void clearRoom()}
          disabled={busy || blocked}
          title={blocked ? 'Wait for the tile render to finish' : undefined}
        >
          {busy ? `Cleaning… ${Math.round(elapsed)}s` : data ? 'Re-run' : 'Clean Room'}
        </button>

        {data?.counts && (
          <span className="analysis-count">
            {data.counts.objects} objects · {data.counts.mirrors} mirrors
            {data.surfaces?.length
              ? ` · ${data.surfaces
                  .map((item) => `${item.name} ${(100 * (item.coverage ?? 0)).toFixed(0)}%`)
                  .join(' · ')}`
              : ''}
          </span>
        )}
      </header>

      {error && (
        <p className="analysis-error" role="alert">
          {error}
        </p>
      )}

      <div className="analysis-strip">
        <figure>
          <img src={room.previewUrl} alt="Original room" />
          <figcaption>Original Room</figcaption>
        </figure>

        <figure>
          {data && objects ? (
            <CleanRoom
              originalUrl={room.previewUrl}
              cleanUrl={data.clean_room_png}
              objectsUrl={objects.cutout_url}
              mirrorsUrl={mirrors?.pixels ? mirrors.cutout_url : undefined}
              filledPx={data.clean_room?.filled_px}
            />
          ) : (
            <span className="analysis-slot">{busy ? 'Cleaning…' : 'Not cleaned yet'}</span>
          )}
          <figcaption>Clean Room</figcaption>
        </figure>

        <figure>
          {objects ? (
            <img className="alpha" src={objects.cutout_url} alt="All objects" />
          ) : (
            <span className="analysis-slot">{busy ? 'Working…' : 'Not cleaned yet'}</span>
          )}
          <figcaption>
            All Objects PNG
            {objects ? <em>{objects.pixels.toLocaleString()} px</em> : null}
          </figcaption>
        </figure>

        <figure>
          {mirrors ? (
            <img className="alpha" src={mirrors.cutout_url} alt="Mirrors" />
          ) : (
            <span className="analysis-slot">{busy ? 'Working…' : 'Not cleaned yet'}</span>
          )}
          <figcaption>
            Mirrors PNG
            {mirrors ? <em>{mirrors.pixels.toLocaleString()} px</em> : null}
          </figcaption>
        </figure>
      </div>

      {/* Floor and wall of the cleaned room. Shown only once the stage has
          produced them, so nothing above changes for a room that has not been
          cleaned or for a backend that does not send these fields. */}
      {data?.floor_mask_png && data?.wall_mask_png && (
        <div className="analysis-strip">
          <figure>
            <img src={data.clean_room_png ?? room.previewUrl} alt="Clean room" />
            <figcaption>Original</figcaption>
          </figure>

          <figure>
            <img src={data.floor_mask_png} alt="Floor mask" />
            <figcaption>
              Floor Mask
              {data.floor_wall?.stats?.floor_coverage !== undefined ? (
                <em>{(100 * data.floor_wall.stats.floor_coverage).toFixed(1)}% of frame</em>
              ) : null}
            </figcaption>
          </figure>

          <figure>
            <img src={data.wall_mask_png} alt="Wall mask" />
            <figcaption>
              Wall Mask
              {data.floor_wall?.stats?.wall_coverage !== undefined ? (
                <em>{(100 * data.floor_wall.stats.wall_coverage).toFixed(1)}% of frame</em>
              ) : null}
            </figcaption>
          </figure>
        </div>
      )}

      <p className="analysis-note">
        {data?.clean_room?.available ? (
          <>
            The clean room is the photograph with both object layers removed and the
            floor and wall behind them rebuilt by LaMa, inside those masks and nowhere
            else — {data.clean_room.checks?.outside_changed ?? 0} pixels outside them
            changed, so the architecture, camera, perspective and lighting of
            everything that was not an object are the original photograph. Tiles are
            projected onto this room, and the two object layers are composited back
            over them at the end, which is why the objects return exactly where they
            were.
          </>
        ) : (
          <>
            The clean room is the photograph with both object layers punched out, so
            every pixel left in it is the original. The chequered gaps are where the
            objects stood: the reconstruction could not run
            {data?.clean_room?.reason ? ` — ${data.clean_room.reason}` : ''}, so nothing
            was repainted behind them. Those same layers are composited back over the
            new tiles at the end, which is why the objects return exactly where they
            were.
          </>
        )}
      </p>
    </section>
  )
}
