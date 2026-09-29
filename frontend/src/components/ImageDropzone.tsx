import { useId, useRef, useState } from 'react'
import type { DragEvent } from 'react'
import type { UploadedImage } from '../types'

interface ImageDropzoneProps {
  label: string
  dropHint: string
  buttonLabel: string
  replaceLabel: string
  image: UploadedImage | null
  error?: string
  onSelect: (file: File) => void
  onClear: () => void
  variant?: 'room' | 'tile'
}

export function ImageDropzone({
  label,
  dropHint,
  buttonLabel,
  replaceLabel,
  image,
  error,
  onSelect,
  onClear,
  variant = 'room',
}: ImageDropzoneProps) {
  const inputId = useId()
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [localError, setLocalError] = useState<string | null>(null)

  function accept(file: File | undefined) {
    if (!file) return

    if (!file.type.startsWith('image/')) {
      setLocalError('That file is not an image. Please choose a JPG or PNG.')
      return
    }

    setLocalError(null)
    onSelect(file)
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault()
    setDragging(false)
    accept(event.dataTransfer.files?.[0])
  }

  function handleDragOver(event: DragEvent<HTMLDivElement>) {
    event.preventDefault()
    setDragging(true)
  }

  const message = error ?? localError

  if (image) {
    return (
      <div className={`dropzone filled ${variant}`}>
        <div className="preview">
          <img src={image.previewUrl} alt={`${label} preview`} />
        </div>
        <div className="preview-meta">
          <p className="file-name" title={image.file.name}>
            {image.file.name}
          </p>
          <p className="file-size">{formatSize(image.file.size)}</p>
          <div className="preview-actions">
            <button
              type="button"
              className="btn subtle"
              onClick={() => inputRef.current?.click()}
            >
              {replaceLabel}
            </button>
            <button type="button" className="btn danger-link" onClick={onClear}>
              Remove
            </button>
          </div>
        </div>
        <input
          ref={inputRef}
          id={inputId}
          type="file"
          accept="image/*"
          className="visually-hidden"
          onChange={(event) => accept(event.target.files?.[0])}
        />
      </div>
    )
  }

  return (
    <>
      <div
        className={`dropzone ${dragging ? 'dragging' : ''} ${message ? 'invalid' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={() => setDragging(false)}
      >
        <div className="dropzone-icon" aria-hidden="true">
          {variant === 'tile' ? <TileIcon /> : <RoomIcon />}
        </div>
        <p className="dropzone-title">{dropHint}</p>
        <p className="dropzone-or">or</p>
        <label htmlFor={inputId} className="btn primary-outline">
          {buttonLabel}
        </label>
        <input
          id={inputId}
          type="file"
          accept="image/*"
          className="visually-hidden"
          onChange={(event) => accept(event.target.files?.[0])}
        />
        <p className="dropzone-formats">JPG or PNG</p>
      </div>
      {message && <p className="field-error">{message}</p>}
    </>
  )
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function RoomIcon() {
  return (
    <svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" strokeWidth="1.6">
      <rect x="3" y="4" width="18" height="16" rx="2" />
      <path d="M3 15l5-4 4 3 3-2 6 4" />
      <circle cx="8.5" cy="8.5" r="1.4" />
    </svg>
  )
}

function TileIcon() {
  return (
    <svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" strokeWidth="1.6">
      <rect x="3" y="3" width="8" height="8" rx="1" />
      <rect x="13" y="3" width="8" height="8" rx="1" />
      <rect x="3" y="13" width="8" height="8" rx="1" />
      <rect x="13" y="13" width="8" height="8" rx="1" />
    </svg>
  )
}
