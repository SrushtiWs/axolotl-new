/**
 * Screen one: choose the room to transform.
 *
 * Either an upload or one of the demo rooms. Both routes end at the same place
 * — a File plus a preview URL — so the studio does not care which was used.
 * A demo room is fetched and turned into a File here rather than being special
 * cased downstream, because POST /generate takes multipart uploads and nothing
 * about a demo room should reach the backend differently from a real photo.
 */

import { useEffect, useMemo, useRef, useState } from 'react'
import { Brand } from '../components/Brand'
import { validateRoomImage } from '../services/roomImage'
import type { Catalogue, DemoRoom, UploadedImage } from '../types'

interface Props {
  catalogue: Catalogue | null
  onRoomChosen: (image: UploadedImage) => void
}

const ALL = 'All'

export function RoomPicker({ catalogue, onRoomChosen }: Props) {
  const [active, setActive] = useState<string>('Living Room')
  const [loading, setLoading] = useState<string | null>(null)
  const [checking, setChecking] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fileInput = useRef<HTMLInputElement | null>(null)

  const categories = useMemo(() => {
    if (!catalogue) return []

    const counts = new Map<string, number>()

    for (const room of catalogue.rooms) {
      counts.set(room.category, (counts.get(room.category) ?? 0) + 1)
    }

    return [
      { name: ALL, count: catalogue.rooms.length },
      ...[...counts.entries()]
        .sort((a, b) => b[1] - a[1])
        .map(([name, count]) => ({ name, count })),
    ]
  }, [catalogue])

  useEffect(() => {
    if (!categories.length) return
    if (!categories.some((item) => item.name === active)) setActive(categories[0].name)
  }, [categories, active])

  const rooms = useMemo(() => {
    if (!catalogue) return []
    return active === ALL
      ? catalogue.rooms
      : catalogue.rooms.filter((room) => room.category === active)
  }, [catalogue, active])

  /**
   * Gate the upload on the instant checks before the studio opens. The
   * backend's floor detection has the final say once the room is cleared;
   * this only stops the obvious mistake — a tile or a product shot dropped
   * into the room field — without a round trip.
   */
  async function useUpload(file: File | null | undefined) {
    if (!file) return

    setError(null)
    setChecking(true)

    const verdict = await validateRoomImage(file)

    setChecking(false)

    if (!verdict.ok) {
      setError(verdict.reason ?? 'That image cannot be used as a room.')
      return
    }

    onRoomChosen({ file, previewUrl: URL.createObjectURL(file) })
  }

  async function useDemo(room: DemoRoom) {
    setLoading(room.id)
    setError(null)

    try {
      const response = await fetch(room.src)

      if (!response.ok) throw new Error(`Could not load ${room.name}.`)

      const blob = await response.blob()
      const file = new File([blob], `${room.id}.jpg`, { type: blob.type || 'image/jpeg' })

      onRoomChosen({ file, previewUrl: URL.createObjectURL(file) })
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not load that room.')
      setLoading(null)
    }
  }

  return (
    <div className="picker">
      <div className="picker-banner">
        Enjoying our visualizer? Start your <a href="#trial">free trial!</a>
      </div>

      <header className="picker-topbar">
        <Brand size="full" />
      </header>

      <section className="picker-hero">
        <div className="picker-hero-copy">
          <h1>Select a room for Transformation</h1>

          <ul className="picker-points">
            <li>
              <span aria-hidden="true">📷</span> Upload a picture of your room
            </li>
            <li>
              <span aria-hidden="true">🧊</span> Try our products in your room
            </li>
          </ul>

          <button
            type="button"
            className="btn primary xl"
            onClick={() => fileInput.current?.click()}
            disabled={checking}
          >
            <span aria-hidden="true">📷</span> {checking ? 'Checking image…' : 'Upload'}
          </button>

          <input
            ref={fileInput}
            type="file"
            accept="image/*"
            hidden
            onChange={(event) => {
              void useUpload(event.target.files?.[0])
              // Clear it so re-picking the same file fires onChange again.
              event.target.value = ''
            }}
          />

          <button type="button" className="btn outline wide">
            <span aria-hidden="true">▦</span> Or scan a QR code to upload pictures
          </button>

          {error && (
            <p className="picker-error" role="alert">
              <strong>⚠</strong> {error}
            </p>
          )}
        </div>

        <div className="picker-hero-art" aria-hidden="true">
          <div className="art-swatches">
            <span /> <span /> <span className="art-active" />
          </div>
          <div className="art-counter">
            <div className="art-top" />
            <div className="art-body">
              <div className="art-lines" />
              <div className="art-panel" />
              <div className="art-drawers">
                <i /> <i /> <i />
              </div>
            </div>
            <div className="art-legs">
              <i /> <i />
            </div>
          </div>
          <span className="art-tick">✓</span>
        </div>
      </section>

      <section className="picker-demos">
        <h2>No picture? Try our demo rooms instead</h2>

        <div className="picker-tabs" role="tablist">
          {['2D', 'Panorama', 'My Rooms'].map((tab, index) => (
            <button
              key={tab}
              type="button"
              role="tab"
              aria-selected={index === 0}
              className={index === 0 ? 'picker-tab active' : 'picker-tab'}
              disabled={index > 0}
              title={index > 0 ? 'Not available in this build' : undefined}
            >
              {tab}
            </button>
          ))}
        </div>

        <div className="picker-chips">
          {categories.map((item) => (
            <button
              key={item.name}
              type="button"
              className={item.name === active ? 'chip active' : 'chip'}
              onClick={() => setActive(item.name)}
            >
              {item.name} ({item.count})
            </button>
          ))}
        </div>

        <div className="picker-grid">
          {rooms.map((room) => (
            <button
              key={room.id}
              type="button"
              className="room-card"
              onClick={() => useDemo(room)}
              disabled={loading !== null}
            >
              <img src={room.thumb} alt={room.name} loading="lazy" />
              <span className="room-card-name">{room.name}</span>
              {loading === room.id && <span className="room-card-loading">Loading…</span>}
            </button>
          ))}

          {!catalogue && <p className="picker-empty">Loading rooms…</p>}
          {catalogue && rooms.length === 0 && (
            <p className="picker-empty">No demo rooms in this category.</p>
          )}
        </div>
      </section>

      <footer className="picker-footer">
        <Brand size="full" />
      </footer>
    </div>
  )
}
