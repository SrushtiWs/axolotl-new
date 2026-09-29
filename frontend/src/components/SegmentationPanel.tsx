/**
 * Segmentation output for the room currently in the form.
 *
 * The extraction produces exactly two images — `ALL_OBJECTS.png` and
 * `MIRRORS_ONLY.png` — each the full size of the room photo, each holding its
 * objects at the photo's own coordinates so they composite straight back over
 * a retiled room. The per-object detections behind them are kept for debugging
 * but are not outputs of their own, and no per-object PNG is written.
 *
 * Upload any photo and it is segmented live on the backend CPU. With no upload
 * yet, the panel shows the production pipeline's own committed props and mirror
 * composed into the same two layers.
 *
 * Every image here is cut from real pixels — nothing is generated.
 */
import { useEffect, useState } from 'react'
import { fetchSegments, isBackendConfigured, segmentRoom } from '../services/api'
import { DownloadButton } from './DownloadButton'
import type { Detection, Segment, SegmentsResponse, UploadedImage } from '../types'

type Tab = 'objects' | 'surfaces'

interface SegmentationPanelProps {
  roomImage: UploadedImage | null
}

function SegmentCard({ segment }: { segment: Segment }) {
  const [x0, y0, x1, y1] = segment.bbox

  return (
    <figure className="segment-card">
      <div className="segment-thumb">
        <img src={segment.cutout_url} alt={segment.name} loading="lazy" />
      </div>

      <figcaption>
        <strong>{segment.name}</strong>
        <span className="segment-meta">{segment.pixels.toLocaleString()} px</span>
        <span className="segment-meta">
          {x1 - x0} × {y1 - y0} at ({x0}, {y0})
        </span>
      </figcaption>
    </figure>
  )
}

function LayerCard({ layer }: { layer: Segment }) {
  const [, , width, height] = layer.bbox

  const count = layer.members?.length ?? 0

  return (
    <figure className="segment-card layer-card">
      <div className="segment-thumb layer-thumb">
        {layer.pixels > 0 ? (
          <img src={layer.cutout_url} alt={layer.name} loading="lazy" />
        ) : (
          <span className="segment-meta">nothing of this kind in the room</span>
        )}
      </div>

      <figcaption>
        <strong>{layer.name}</strong>
        <span className="segment-meta">
          {layer.filename ?? `${layer.id}.png`} · {width} × {height}
        </span>
        <span className="segment-meta">
          {count} object{count === 1 ? '' : 's'} · {layer.pixels.toLocaleString()} px
        </span>

        <DownloadButton
          url={layer.cutout_url ?? ''}
          filename={layer.filename ?? `${layer.id}.png`}
          // An empty mirrors layer is a valid result but not worth saving, and
          // a surface entry carries no cut-out at all.
          disabled={layer.pixels === 0 || !layer.cutout_url}
        />
      </figcaption>
    </figure>
  )
}

function DetectionList({ detections }: { detections: Detection[] }) {
  const kept = detections.filter((item) => item.status === 'object' || item.status === 'mirror')

  return (
    <details className="detection-detail">
      <summary>
        {kept.length} of {detections.length} detections used
      </summary>

      <ul className="detection-list">
        {detections.map((item, index) => (
          <li key={`${item.name}-${index}`} className={`detection detection-${item.status}`}>
            <span className="detection-name">{item.name}</span>
            <span className="segment-meta">
              {item.status}
              {item.pixels ? ` · ${item.pixels.toLocaleString()} px` : ''}
              {item.reason ? ` · ${item.reason}` : ''}
            </span>
          </li>
        ))}
      </ul>
    </details>
  )
}

export function SegmentationPanel({ roomImage }: SegmentationPanelProps) {
  const [data, setData] = useState<SegmentsResponse | null>(null)
  const [tab, setTab] = useState<Tab>('objects')
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const file = roomImage?.file ?? null

  useEffect(() => {
    if (!isBackendConfigured()) {
      setError('No backend connected, so no segmentation output can be shown.')
      return
    }

    const controller = new AbortController()

    setLoading(true)
    setError(null)
    setData(null)

    // An uploaded photo is segmented on the spot; otherwise fall back to the
    // pipeline's committed segmentation of the calibrated room.
    const request = file
      ? segmentRoom(file, controller.signal)
      : fetchSegments(controller.signal)

    request
      .then(setData)
      .catch((cause) => {
        if (controller.signal.aborted) return
        setError(cause instanceof Error ? cause.message : 'Could not segment this room.')
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })

    return () => controller.abort()
  }, [file])

  const segments = data ? (tab === 'objects' ? data.objects : data.surfaces) : []

  const isLive = data?.source === 'live'

  return (
    <aside className="pipeline-info">
      <div className="segment-head">
        <h3>Room segmentation</h3>

        {data && (
          <div className="segment-tabs" role="tablist">
            {(['objects', 'surfaces'] as Tab[]).map((value) => (
              <button
                key={value}
                type="button"
                role="tab"
                aria-selected={tab === value}
                className={`segment-tab ${tab === value ? 'selected' : ''}`}
                onClick={() => setTab(value)}
              >
                {value === 'objects'
                  ? `Object layers (${data.objects.length})`
                  : `Surfaces (${data.surfaces.length})`}
              </button>
            ))}
          </div>
        )}
      </div>

      {loading && (
        <p className="pipeline-note">
          {file ? 'Segmenting your room on the CPU — this takes a few seconds…' : 'Loading…'}
        </p>
      )}

      {error && <p className="pipeline-note">{error}</p>}

      {data && segments.length === 0 && !loading && (
        <p className="pipeline-note">
          {tab === 'objects'
            ? 'No objects were found in this photo above the minimum size.'
            : 'No wall, floor or ceiling was identified in this photo.'}
        </p>
      )}

      {segments.length > 0 && (
        <div className={`segment-grid${tab === 'objects' ? ' layer-grid' : ''}`}>
          {segments.map((segment) =>
            tab === 'objects' ? (
              <LayerCard key={segment.id} layer={segment} />
            ) : (
              <SegmentCard key={segment.id} segment={segment} />
            ),
          )}
        </div>
      )}

      {tab === 'objects' && data?.detections && data.detections.length > 0 && (
        <DetectionList detections={data.detections} />
      )}

      {data && !loading && (
        <p className="pipeline-note">
          {isLive
            ? 'Two layers cut from your uploaded photo — SegFormer-B4 proposes, SAM draws each boundary, a guided-filter matte softens the edge. Both are the full size of your photo, with every object left where it was, so they composite straight back over a retiled room.'
            : "The production pipeline's committed props and mirror, composed into the same two layers. Upload a room image to extract from that photo instead."}
        </p>
      )}
    </aside>
  )
}
