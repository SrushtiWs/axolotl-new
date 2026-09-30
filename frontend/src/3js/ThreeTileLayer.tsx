/**
 * The optional 3D view of the selected surfaces' tiles.
 *
 * Stacked in the image's own box, exactly like the 2D composite:
 *   the photograph -> the tiles (Three.js, clipped to each surface's mask)
 *   -> the room's objects, from the photograph, over the tiles.
 *
 * It only consumes what each render recorded (`three.json`); the 2D image
 * remains the default view and is untouched.
 */

import { useEffect, useRef, useState } from 'react'

import { API_BASE_URL } from '../services/api'
import type { ImageFrame } from '../components/SurfaceMarkers'
import { loadThreeJob, roomConsistency, TileLayerRenderer } from './threeTiles'
import type { ThreeJobData, ThreeLayer } from './threeTiles'
import './three-layer.css'

interface ThreeTileLayerProps {
  /** The surfaces on screen, each with the render job that tiled it. */
  layers: { surface: string; job: string }[]
  /** Where the image sits inside the stage. */
  frame: ImageFrame | null
  /** Reports why the view cannot be shown (e.g. a render from before 3D data existed). */
  onUnavailable?: (reason: string) => void
}

const jobBase = (job: string) => `${API_BASE_URL}/jobs/${job}/`

export function ThreeTileLayer({ layers, frame, onUnavailable }: ThreeTileLayerProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const rendererRef = useRef<TileLayerRenderer | null>(null)
  const [first, setFirst] = useState<ThreeJobData | null>(null)
  const [firstJob, setFirstJob] = useState<string | null>(null)
  const [ready, setReady] = useState(false)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const renderer = new TileLayerRenderer(canvas)
    rendererRef.current = renderer
    return () => {
      renderer.dispose()
      rendererRef.current = null
    }
  }, [])

  const key = layers.map((item) => `${item.job}:${item.surface}`).join(',')

  useEffect(() => {
    const controller = new AbortController()
    setReady(false)

    Promise.all(
      layers.map(async (item): Promise<ThreeLayer> => ({
        jobBase: jobBase(item.job),
        surface: item.surface,
        data: await loadThreeJob(`${jobBase(item.job)}three.json`),
      })),
    )
      .then(async (resolved) => {
        if (controller.signal.aborted) return
        const missing = resolved.find((item) => !item.data.surfaces[item.surface])
        if (missing) throw new Error(`no 3D data for ${missing.surface}`)
        // Only the room object's own camera, scale and tile mm are drawn.
        for (const item of resolved) {
          const problems = roomConsistency(item.data, item.surface)
          if (problems.length) throw new Error(`${item.surface} does not match its room (${problems.join('; ')})`)
        }
        await rendererRef.current?.render(resolved)
        if (controller.signal.aborted) return
        setFirst(resolved[0]?.data ?? null)
        setFirstJob(layers[0]?.job ?? null)
        setReady(true)
      })
      .catch((cause) => {
        if (controller.signal.aborted) return
        onUnavailable?.(
          cause instanceof Error
            ? `3D view unavailable: ${cause.message}. Re-select the surface to render it again.`
            : '3D view unavailable.',
        )
      })

    return () => controller.abort()
    // `layers` is fully described by `key`.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])

  const box = frame
    ? { left: frame.left, top: frame.top, width: frame.width, height: frame.height }
    : { left: 0, top: 0, width: '100%', height: '100%' }

  return (
    <div className="three-layer" style={{ ...box, visibility: ready ? 'visible' : 'hidden' }} aria-label="3D tile view">
      {first && firstJob && <img src={jobBase(firstJob) + first.photo} alt="" draggable={false} />}
      <canvas ref={canvasRef} data-testid="three-tiles" />
      {first && firstJob && <img src={jobBase(firstJob) + first.objects} alt="" draggable={false} />}
    </div>
  )
}
