/**
 * Before / after wipe.
 *
 * Both images are drawn at the same size and the top one is clipped, so the
 * two halves always line up pixel for pixel — the tiled render is the original
 * photo's exact dimensions, which is what makes a straight clip correct here.
 */

import { useRef, useState } from 'react'

interface Props {
  before: string
  after: string
  onClose?: () => void
}

export function CompareSlider({ before, after, onClose }: Props) {
  const [at, setAt] = useState(50)
  const frame = useRef<HTMLDivElement | null>(null)

  function move(clientX: number) {
    const box = frame.current?.getBoundingClientRect()
    if (!box || box.width === 0) return
    setAt(Math.min(100, Math.max(0, ((clientX - box.left) / box.width) * 100)))
  }

  return (
    <div
      ref={frame}
      className="compare"
      onPointerDown={(event) => {
        event.currentTarget.setPointerCapture(event.pointerId)
        move(event.clientX)
      }}
      onPointerMove={(event) => {
        if (event.buttons === 1) move(event.clientX)
      }}
    >
      <img className="compare-img" src={before} alt="Original room" draggable={false} />

      <div className="compare-top" style={{ clipPath: `inset(0 ${100 - at}% 0 0)` }}>
        <img className="compare-img" src={after} alt="Tiled room" draggable={false} />
      </div>

      <div className="compare-handle" style={{ left: `${at}%` }}>
        <span>◂▸</span>
      </div>

      <span className="compare-tag left">After</span>
      <span className="compare-tag right">Before</span>

      <div className="compare-bar" onPointerDown={(event) => event.stopPropagation()}>
        <button type="button" onClick={() => setAt(100)} title="Show the tiled room">
          Left
        </button>
        <button type="button" onClick={() => onClose?.()} title="Close compare">
          ✕
        </button>
        <button type="button" onClick={() => setAt(0)} title="Show the original photo">
          Right
        </button>
      </div>
    </div>
  )
}
