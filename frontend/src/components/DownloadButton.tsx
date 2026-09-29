/**
 * Saves a backend image to disk under a chosen filename.
 *
 * The obvious implementation — `<a href={url} download="ALL_OBJECTS.png">` —
 * does not work here. The images are served by the backend on port 8000 while
 * this app runs on 5173, and browsers ignore the `download` attribute on a
 * cross-origin URL: the link navigates to the image instead of saving it, and
 * whatever filename was asked for is lost.
 *
 * So the bytes are fetched first (the backend already allows this origin in its
 * CORS policy), turned into a blob: URL — which is same-origin, and therefore
 * honours both `download` and the filename — and clicked.
 */
import { useEffect, useRef, useState } from 'react'

interface DownloadButtonProps {
  /** Absolute URL of the image to save. */
  url: string
  /** Filename offered to the user, e.g. `ALL_OBJECTS.png`. */
  filename: string
  label?: string
  className?: string
  /** Set when there is nothing worth saving, e.g. an empty mirrors layer. */
  disabled?: boolean
}

type State = 'idle' | 'saving' | 'error'

export function DownloadButton({
  url,
  filename,
  label = 'Download PNG',
  className = 'btn subtle download-btn',
  disabled = false,
}: DownloadButtonProps) {
  const [state, setState] = useState<State>('idle')

  // A save that finishes after the panel has moved on must not set state on an
  // unmounted component, and a pending object URL must still be released.
  const alive = useRef(true)
  const pending = useRef<number[]>([])

  useEffect(() => {
    alive.current = true

    return () => {
      alive.current = false
      pending.current.forEach((id) => window.clearTimeout(id))
    }
  }, [])

  async function save() {
    if (disabled || state === 'saving') return

    setState('saving')

    try {
      const response = await fetch(url)

      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`)
      }

      const blob = await response.blob()

      const objectUrl = URL.createObjectURL(blob)

      const anchor = document.createElement('a')
      anchor.href = objectUrl
      anchor.download = filename

      document.body.appendChild(anchor)
      anchor.click()
      anchor.remove()

      // Revoking straight away can cancel the save while the browser is still
      // reading the blob, so the URL is released on a delay instead.
      pending.current.push(
        window.setTimeout(() => URL.revokeObjectURL(objectUrl), 60_000),
      )

      if (alive.current) setState('idle')
    } catch {
      if (alive.current) setState('error')
    }
  }

  return (
    <button
      type="button"
      className={className}
      onClick={save}
      disabled={disabled || state === 'saving'}
      title={disabled ? 'Nothing to download' : `Save ${filename}`}
    >
      {state === 'saving' ? 'Saving…' : state === 'error' ? 'Retry download' : label}
    </button>
  )
}
