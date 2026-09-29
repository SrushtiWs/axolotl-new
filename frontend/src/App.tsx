/**
 * Two screens, one piece of state: which room is being worked on.
 *
 * No router — the choice of screen *is* whether a room has been picked, so a
 * URL would only be able to say the same thing twice.
 */

import { useEffect, useState } from 'react'
import { RoomPicker } from './pages/RoomPicker'
import { Studio } from './pages/Studio'
import type { Catalogue, UploadedImage } from './types'

export default function App() {
  const [catalogue, setCatalogue] = useState<Catalogue | null>(null)
  const [room, setRoom] = useState<UploadedImage | null>(null)

  useEffect(() => {
    let live = true

    fetch('/demo/catalogue.json')
      .then((response) => (response.ok ? response.json() : Promise.reject(response.status)))
      .then((data: Catalogue) => {
        if (live) setCatalogue(data)
      })
      .catch(() => {
        // The catalogue is demo content, not a dependency: without it the
        // upload route still works, so this fails quietly rather than
        // blocking the screen.
        if (live) setCatalogue({ rooms: [], tiles: [] })
      })

    return () => {
      live = false
    }
  }, [])

  if (!room) {
    return <RoomPicker catalogue={catalogue} onRoomChosen={setRoom} />
  }

  return (
    <Studio
      catalogue={catalogue}
      room={room}
      onChangeRoom={() => {
        URL.revokeObjectURL(room.previewUrl)
        setRoom(null)
      }}
    />
  )
}
