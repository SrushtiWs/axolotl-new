/**
 * The room with its objects taken out and the room behind them put back.
 *
 * The backend now reconstructs those pixels — LaMa, inside the object masks and
 * nowhere else — and returns `CLEAN_ROOM.png`, an ordinary opaque photograph of
 * the emptied room. When that image is there it is simply shown: it is the
 * real output of the pipeline, and redrawing it here would only risk
 * disagreeing with what the renderer actually tiled.
 *
 * The canvas path below is the fallback for when the fill could not run — a
 * missing weight file, an inpainting failure. It composes the original minus
 * both object layers with `destination-out`, which leaves transparent gaps
 * where the objects stood. That is visibly not a finished room, which is the
 * point: it shows what is missing rather than implying a reconstruction that
 * did not happen.
 *
 * Nothing is read back out of the canvas, only displayed, so a cross-origin
 * layer tainting it costs nothing but the pixel count.
 */

import { useEffect, useRef, useState } from 'react'

interface Props {
  originalUrl: string
  /** `CLEAN_ROOM.png` from the backend, when the fill ran. */
  cleanUrl?: string | null
  objectsUrl?: string
  mirrorsUrl?: string
  /** Pixels the backend reconstructed, for the caption. */
  filledPx?: number
}

function load(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image()
    image.crossOrigin = 'anonymous'
    image.onload = () => resolve(image)
    image.onerror = () => reject(new Error(`Could not load ${src}`))
    image.src = src
  })
}

export function CleanRoom({ originalUrl, cleanUrl, objectsUrl, mirrorsUrl, filledPx }: Props) {
  const canvas = useRef<HTMLCanvasElement | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [removed, setRemoved] = useState<number | null>(null)

  useEffect(() => {
    // The reconstructed room needs no compositing.
    if (cleanUrl) return

    let live = true

    async function compose() {
      setError(null)
      setRemoved(null)

      try {
        const layers = [objectsUrl, mirrorsUrl].filter(Boolean) as string[]

        const [base, ...cutouts] = await Promise.all([
          load(originalUrl),
          ...layers.map(load),
        ])

        if (!live) return

        const element = canvas.current
        if (!element) return

        element.width = base.naturalWidth
        element.height = base.naturalHeight

        const context = element.getContext('2d')
        if (!context) return

        context.clearRect(0, 0, element.width, element.height)
        context.drawImage(base, 0, 0)

        // Erase wherever a layer is opaque: the objects, and only the objects.
        context.globalCompositeOperation = 'destination-out'

        for (const layer of cutouts) {
          context.drawImage(layer, 0, 0, element.width, element.height)
        }

        context.globalCompositeOperation = 'source-over'

        // How much of the room was lifted out, for the caption.
        try {
          const data = context.getImageData(0, 0, element.width, element.height).data
          let clear = 0

          for (let i = 3; i < data.length; i += 4) {
            if (data[i] === 0) clear += 1
          }

          if (live) setRemoved(clear)
        } catch {
          // A tainted canvas blocks the count, not the picture.
        }
      } catch (cause) {
        if (live) setError(cause instanceof Error ? cause.message : 'Could not build the clean room.')
      }
    }

    void compose()

    return () => {
      live = false
    }
  }, [originalUrl, cleanUrl, objectsUrl, mirrorsUrl])

  if (cleanUrl) {
    return (
      <>
        <img className="clean-image" src={cleanUrl} alt="Clean room" />
        {filledPx !== undefined && (
          <span className="clean-meta">{filledPx.toLocaleString()} px reconstructed</span>
        )}
      </>
    )
  }

  return (
    <>
      <canvas ref={canvas} className="clean-canvas" aria-label="Clean room" />
      {error && <span className="analysis-slot">{error}</span>}
      {removed !== null && (
        <span className="clean-meta">{removed.toLocaleString()} px removed, not refilled</span>
      )}
    </>
  )
}
