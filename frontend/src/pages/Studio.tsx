/**
 * Screen two: the tiling studio.
 *
 * Left rail is the product catalogue, the middle is the room with its controls
 * stacked beside it, and below sit the three shelves — Saved Designs, Saved
 * Rooms and Surfaces.
 *
 * Surfaces are chosen on the room itself. As soon as a room opens it is
 * cleaned, and every surface the tile engine finds gets a control: a "Floor"
 * pill, and a dot on each wall however many there are.
 *
 *   select a surface     it takes the current tile, straight away, and follows
 *                        every tile / Layout / Grout / Room Size change after
 *   deselect it          it keeps the tiles it has; later changes pass it by
 *   select it again      it takes whatever tile is current then
 *
 * So the floor and each wall can each carry their own tile. There is no Apply.
 *
 * How that stays exact and glitch-free: every surface is rendered on its own —
 * the floor, or one wall as its own scale anchor (POST /generate with
 * `wall_id`) — once per set of settings, and what is shown is POST /compose:
 * the photograph with each surface's tiles copied in from its own render,
 * inside its own region only. A surface keeps showing its current tiles until
 * the new ones are rendered and loaded, so nothing blinks or disappears.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Brand } from '../components/Brand'
import { CompareSlider } from '../components/CompareSlider'
import { RoomAnalysis } from '../components/RoomAnalysis'
import { SurfaceMarkers } from '../components/SurfaceMarkers'
import { ThreeTileLayer } from '../3js'
import type { ImageFrame, SurfaceMarker } from '../components/SurfaceMarkers'
import { TileRail } from '../components/TileRail'
import {
  API_BASE_URL,
  composeLayers,
  fetchSurfaces,
  generateVisualization,
  isBackendConfigured,
} from '../services/api'
import { ROTATIONS } from '../types'
import type {
  CatalogueTile,
  Catalogue,
  ComposeResponse,
  RoomMode,
  Rotation,
  SegmentsResponse,
  SurfacesResponse,
  UploadedImage,
} from '../types'

interface Props {
  catalogue: Catalogue | null
  room: UploadedImage
  onChangeRoom: () => void
}

type Panel = 'grout' | 'layout' | 'size' | null

interface SavedDesign {
  id: string
  url: string
  tile: string
  /** Which surfaces were tiled, e.g. "Floor, Wall 2". */
  surface: string
  /** "Room 1.2" — which room, and which variant of it. */
  label?: string
}

interface SavedRoom {
  id: string
  url: string
  name: string
}

/** Everything one render depends on, frozen when a surface is given it. */
interface Settings {
  tileName: string
  tileFile: () => Promise<File | null>
  rotation: Rotation
  grout: number
  size: { width: string; length: string; height: string }
  tileSize: { width: number; height: number }
}

type RoomSize = { width: string; length: string; height: string }

/** The room dimensions the user typed, in feet. Empty fields are left out. */
function typedRoomSize(size: RoomSize) {
  const typed: { room_width?: number; room_length?: number; room_height?: number } = {}
  if (size.width.trim()) typed.room_width = Number(size.width)
  if (size.length.trim()) typed.room_length = Number(size.length)
  if (size.height.trim()) typed.room_height = Number(size.height)
  return typed
}

/** MANUAL as soon as any dimension is typed; AUTO while all are empty. */
function roomModeOf(size: RoomSize): RoomMode {
  return size.width.trim() || size.length.trim() || size.height.trim() ? 'manual' : 'auto'
}

/** Grout range in millimetres, and the step the −/+ control moves by. */
const GROUT_MIN = 1
const GROUT_MAX = 12
const GROUT_STEP = 1

/** How long a grout, layout, size or tile change settles before it re-renders. */
const SETTLE_MS = 600

const kindOf = (id: string): 'floor' | 'wall' => (id === 'floor' ? 'floor' : 'wall')

/** A copy of `record` without `key`. */
function without<T>(record: Record<string, T>, key: string): Record<string, T> {
  const { [key]: _dropped, ...rest } = record
  return rest
}

export function Studio({ catalogue, room, onChangeRoom }: Props) {
  const [tile, setTile] = useState<CatalogueTile | null>(null)
  const [upload, setUpload] = useState<UploadedImage | null>(null)

  const [rotation, setRotation] = useState<Rotation>(0)
  const [grout, setGrout] = useState(5)
  // Empty = AUTO (estimated from the photo); any typed value = MANUAL.
  const [size, setSize] = useState({ width: '', length: '', height: '' })
  const roomMode = roomModeOf(size)

  const [panel, setPanel] = useState<Panel>(null)
  const [compare, setCompare] = useState(false)
  // Optional 3D view of the same tiles (Three.js), over the 2D image.
  const [view3d, setView3d] = useState(false)

  /** Before -> Mask -> After: the photo, the white mask tiles may go on, the result. */
  const [step, setStep] = useState<'before' | 'mask' | 'after'>('after')
  const [maskUrl, setMaskUrl] = useState<string | null>(null)

  const [error, setError] = useState<string | null>(null)

  /**
   * Set when the backend's segmentation says this photograph is not a room.
   * Until the room is cleared it is null, and the surface controls stay live —
   * the instant checks in the picker have already passed by then.
   */
  const [invalid, setInvalid] = useState<string | null>(null)

  /** What the clearing pass found. Null until Clean Room has succeeded. */
  const [cleared, setCleared] = useState<SegmentsResponse | null>(null)

  /** True while Clean Room is running. Tiles wait for it: both are CPU jobs,
   *  and run together each takes several times longer — measured at 457s for
   *  a detection that normally costs 80s. */
  const [cleaning, setCleaning] = useState(false)

  /** The surfaces this room offers, each with its control. From POST /surfaces. */
  const [found, setFound] = useState<SurfacesResponse | null>(null)
  const [detecting, setDetecting] = useState(false)

  /** The selected — active — surfaces: "floor", "wall-0", "wall-3"… */
  const [selected, setSelected] = useState<string[]>([])

  /** The settings each surface should carry, by settings key. */
  const [assigned, setAssigned] = useState<Record<string, string>>({})

  /** The settings each surface is showing right now. */
  const [shownKey, setShownKey] = useState<Record<string, string>>({})

  /** Finished renders: `${surface}|${settings key}` -> /generate job id. */
  const [renders, setRenders] = useState<Record<string, string>>({})

  /** Each finished render's notes, by job id. */
  const [jobNotes, setJobNotes] = useState<Record<string, string[]>>({})

  /** Each finished render's room-size check line, and the room size it was for. */
  const [roomChecks, setRoomChecks] = useState<Record<string, { line: string | null; size: string }>>({})

  /** Surfaces being rendered right now. */
  const [pendingIds, setPendingIds] = useState<Record<string, boolean>>({})

  /** Surfaces the engine could not tile, with its reason. */
  const [failures, setFailures] = useState<Record<string, string>>({})

  /** What is on screen. Null shows the photograph itself. */
  const [composed, setComposed] = useState<ComposeResponse | null>(null)

  const [designs, setDesigns] = useState<SavedDesign[]>([])
  const [rooms, setRooms] = useState<SavedRoom[]>([])

  /**
   * Which room this is, for labelling. Designs are numbered inside it —
   * Room 1.1, Room 1.2, Room 1.3 — so a shelf full of variants still says
   * which room each one belongs to.
   */
  // Starts at 0 because the effect below fires on mount too, making the first
  // room Room 1 rather than Room 2.
  const [roomNo, setRoomNo] = useState(0)

  // A new room starts its own numbering and has nothing laid on it yet.
  useEffect(() => {
    setDesigns([])
    setRoomNo((n) => n + 1)
    // Keyed on the room itself; `designs` must not retrigger this.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [room.previewUrl])

  const tileInput = useRef<HTMLInputElement | null>(null)

  useEffect(() => {
    if (!tile && !upload && catalogue?.tiles.length) setTile(catalogue.tiles[0])
  }, [catalogue, tile, upload])

  const activeName = upload ? upload.file.name : (tile?.name ?? 'No tile selected')
  const activeThumb = upload?.previewUrl ?? tile?.thumb ?? null

  const activeSize = useMemo(
    () => (tile ? { width: tile.width, height: tile.height } : { width: 600, height: 1200 }),
    [tile],
  )

  const jobId = cleared?.job_id ?? null

  // An invalid room must not reach the tile pipeline at all.
  const blocked = invalid !== null

  // ------------------------------------------------------------ the surfaces
  //
  // As soon as a room is cleaned, ask which surfaces it offers. A new room, or
  // a re-run of Clean Room, starts from nothing: a render belongs to one
  // cleaned room and must never be shown on another.
  useEffect(() => {
    setFound(null)
    setSelected([])
    setAssigned({})
    setShownKey({})
    setRenders({})
    setJobNotes({})
    setRoomChecks({})
    setPendingIds({})
    setFailures({})
    setComposed(null)
    setError(null)
    inflight.current = {}

    if (!jobId || !isBackendConfigured()) return

    const controller = new AbortController()

    setDetecting(true)

    fetchSurfaces(room.file, jobId, controller.signal)
      .then(setFound)
      .catch((cause) => {
        if (controller.signal.aborted) return
        setError(cause instanceof Error ? cause.message : 'Could not detect surfaces.')
      })
      .finally(() => {
        if (!controller.signal.aborted) setDetecting(false)
      })

    return () => controller.abort()
  }, [jobId, room.file])

  // ------------------------------------------------------------ the settings
  //
  // One key per combination of tile, layout, grout and room size. A surface is
  // assigned a key; the key's settings are frozen here, so a surface that was
  // deselected keeps rendering — and showing — exactly what it was given.
  const settingsKey = JSON.stringify([
    upload ? upload.previewUrl : tile?.id,
    activeSize.width,
    activeSize.height,
    rotation,
    grout,
    size.width,
    size.length,
    size.height,
  ])

  const settingsRef = useRef<Record<string, Settings>>({})

  if (!settingsRef.current[settingsKey]) {
    const frozenUpload = upload
    const frozenTile = tile

    settingsRef.current[settingsKey] = {
      tileName: activeName,
      tileFile: async () => {
        if (frozenUpload) return frozenUpload.file
        if (!frozenTile) return null
        const response = await fetch(frozenTile.src)
        const blob = await response.blob()
        return new File([blob], `${frozenTile.id}.jpg`, { type: blob.type || 'image/jpeg' })
      },
      rotation,
      grout,
      size: { ...size },
      tileSize: { ...activeSize },
    }
  }

  const selectedRef = useRef<string[]>([])
  selectedRef.current = selected

  // A settings change goes to the selected surfaces only, once it settles —
  // a slider drag is one render, not twenty. Deselected surfaces keep theirs.
  const lastKey = useRef(settingsKey)

  useEffect(() => {
    if (lastKey.current === settingsKey) return

    const timer = window.setTimeout(() => {
      lastKey.current = settingsKey

      setAssigned((current) => {
        const next = { ...current }
        for (const id of selectedRef.current) next[id] = settingsKey
        return next
      })
    }, SETTLE_MS)

    return () => window.clearTimeout(timer)
  }, [settingsKey])

  // ------------------------------------------------------------ rendering
  //
  // Every surface whose assigned settings have no render yet is rendered on
  // its own. `inflight` holds the key each surface is being rendered with, so
  // an answer that arrives after the surface was given newer settings is kept
  // for later but never shown in place of the newer one.
  const inflight = useRef<Record<string, string>>({})

  const renderSurface = useCallback(
    async (id: string, key: string) => {
      const settings = settingsRef.current[key]

      if (!jobId || !settings) return

      inflight.current[id] = key
      setPendingIds((current) => ({ ...current, [id]: true }))

      try {
        const tileFile = await settings.tileFile()

        if (!tileFile) throw new Error('Choose a tile first.')

        const response = await generateVisualization({
          room_image: room.file,
          tile_image: tileFile,
          ...typedRoomSize(settings.size),
          room_mode: roomModeOf(settings.size),
          tile_width: settings.tileSize.width,
          tile_height: settings.tileSize.height,
          rotation: settings.rotation,
          surface: kindOf(id),
          grout_mm: settings.grout,
          // This room's Clean Room run: its masks are what the tiles follow.
          job_id: jobId,
          // A wall is rendered on its own, as its own scale anchor.
          wall_id: kindOf(id) === 'wall' ? id : undefined,
        })

        const tiled = (response.surfaces ?? []).some((item) => item.id === id)

        if (!tiled || !response.job_id) {
          throw new Error('the tile engine found no pixels of it to tile')
        }

        // Kept even if superseded: selecting these settings again is instant.
        setRenders((current) => ({ ...current, [`${id}|${key}`]: response.job_id! }))
        setJobNotes((current) => ({ ...current, [response.job_id!]: response.notes ?? [] }))
        const measured = response.geometry?.room as { room_check?: string | null } | undefined
        setRoomChecks((current) => ({
          ...current,
          [response.job_id!]: { line: measured?.room_check ?? null, size: JSON.stringify(settings.size) },
        }))
        setFailures((current) => without(current, id))
      } catch (cause) {
        if (inflight.current[id] !== key) return

        const reason = cause instanceof Error ? cause.message : 'the render failed'

        // Not left selected with nothing on it: it keeps whatever it showed
        // before, stops following changes, and its control says why.
        setFailures((current) => ({ ...current, [id]: reason }))
        setSelected((current) => current.filter((item) => item !== id))
        setAssigned((current) => {
          const shown = shownKeyRef.current[id]
          return shown ? { ...current, [id]: shown } : without(current, id)
        })
      } finally {
        if (inflight.current[id] === key) {
          delete inflight.current[id]
          setPendingIds((current) => without(current, id))
        }
      }
    },
    [jobId, room.file],
  )

  useEffect(() => {
    if (!jobId || blocked || cleaning || !isBackendConfigured()) return

    for (const [id, key] of Object.entries(assigned)) {
      if (renders[`${id}|${key}`] || inflight.current[id] === key) continue
      void renderSurface(id, key)
    }
  }, [jobId, blocked, cleaning, assigned, renders, renderSurface])

  // A surface shows its assigned settings as soon as they are rendered, and
  // keeps showing the previous ones until then.
  const shownKeyRef = useRef<Record<string, string>>({})
  shownKeyRef.current = shownKey

  useEffect(() => {
    setShownKey((current) => {
      let changed = false
      const next = { ...current }

      for (const [id, key] of Object.entries(assigned)) {
        if (renders[`${id}|${key}`] && next[id] !== key) {
          next[id] = key
          changed = true
        }
      }

      return changed ? next : current
    })
  }, [assigned, renders])

  // ------------------------------------------------------------ what is shown
  //
  // The photograph with every surface's tiles, each from its own render. The
  // new image is fully loaded before it replaces the old one, so the room
  // never flashes bare between two states.
  const layers = useMemo(
    () =>
      Object.entries(shownKey)
        .map(([id, key]) => ({ surface: id, job: renders[`${id}|${key}`] }))
        .filter((item): item is { surface: string; job: string } => Boolean(item.job))
        .sort((a, b) => a.surface.localeCompare(b.surface)),
    [shownKey, renders],
  )

  const layersKey = layers.map((item) => `${item.job}:${item.surface}`).join(',')

  useEffect(() => {
    if (!layers.length) {
      setComposed(null)
      return
    }

    const controller = new AbortController()

    composeLayers(layers, controller.signal)
      .then(async (response) => {
        const image = new Image()
        image.src = response.result_image_url
        await image.decode().catch(() => undefined)
        if (!controller.signal.aborted) setComposed(response)
      })
      .catch((cause) => {
        if (controller.signal.aborted) return
        setError(cause instanceof Error ? cause.message : 'Could not show the tiles.')
      })

    return () => controller.abort()
    // `layers` is fully described by `layersKey`.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layersKey])

  // ------------------------------------------------------------ the controls

  /**
   * Select or deselect one surface. Selecting puts the current tile on it at
   * once; deselecting keeps the tiles it has. Nothing else is touched.
   */
  function toggleSurface(marker: SurfaceMarker) {
    const id = marker.id

    if (selected.includes(id)) {
      setSelected((current) => current.filter((item) => item !== id))
      return
    }

    setSelected((current) => [...current, id])
    setFailures((current) => without(current, id))
    setAssigned((current) => ({ ...current, [id]: settingsKey }))
  }

  function changeGrout(next: number) {
    setGrout(Math.min(GROUT_MAX, Math.max(GROUT_MIN, next)))
  }

  function changeRotation(next: Rotation) {
    setRotation(next)
  }

  function reset() {
    setSelected([])
    setAssigned({})
    setShownKey({})
    setFailures({})
    setComposed(null)
    setCompare(false)
    setView3d(false)
    setError(null)
    setRotation(0)
    setGrout(5)
    setPanel(null)
    inflight.current = {}
    setPendingIds({})
  }

  /** The tile a surface is showing, if any. */
  const tileOn = (id: string): string | null =>
    shownKey[id] ? (settingsRef.current[shownKey[id]]?.tileName ?? null) : null

  /** The selected surfaces' names, in the order they are listed on the room. */
  const selectedLabels = (found?.surfaces ?? [])
    .filter((item) => selected.includes(item.id))
    .map((item) => item.label)

  /** Every surface that is showing tiles, by name. */
  const tiledLabels = (found?.surfaces ?? [])
    .filter((item) => shownKey[item.id])
    .map((item) => item.label)

  function saveDesign() {
    if (!composed) return
    setDesigns((current) => [
      {
        id: `d${current.length + 1}`,
        url: composed.result_image_url,
        tile: activeName,
        surface: tiledLabels.join(', '),
        label: `Room ${roomNo}.${current.length + 1}`,
      },
      ...current,
    ])
  }

  function saveRoom() {
    setRooms((current) =>
      current.some((item) => item.url === room.previewUrl)
        ? current
        : [
            { id: `r${current.length + 1}`, url: room.previewUrl, name: `Room ${roomNo}` },
            ...current,
          ],
    )
  }

  /** One control per surface, where the backend measured that surface to be. */
  const markers: SurfaceMarker[] = useMemo(() => {
    if (!found || blocked) return []

    const [width, height] = found.canvas

    return found.surfaces.map((item) => ({
      id: item.id,
      label: item.label,
      x: (100 * item.dot[0]) / width,
      y: (100 * item.dot[1]) / height,
      selected: selected.includes(item.id),
      pending: Boolean(pendingIds[item.id]),
      shape: item.kind === 'floor' ? ('pill' as const) : ('dot' as const),
      failed: failures[item.id],
    }))
  }, [found, blocked, selected, pendingIds, failures])

  // Where the image sits inside the stage. It is letterboxed, so the controls
  // are laid over the image's own box rather than the whole stage.
  const canvasRef = useRef<HTMLDivElement | null>(null)
  const imageRef = useRef<HTMLImageElement | null>(null)
  const [frame, setFrame] = useState<ImageFrame | null>(null)

  const measure = useCallback(() => {
    const canvas = canvasRef.current
    const image = imageRef.current

    if (!canvas || !image) return

    const outer = canvas.getBoundingClientRect()
    const inner = image.getBoundingClientRect()

    setFrame({
      left: inner.left - outer.left,
      top: inner.top - outer.top,
      width: inner.width,
      height: inner.height,
    })
  }, [])

  useEffect(() => {
    const canvas = canvasRef.current

    if (!canvas || typeof ResizeObserver === 'undefined') return

    const observer = new ResizeObserver(measure)
    observer.observe(canvas)

    return () => observer.disconnect()
  }, [measure])

  const tiling = Object.keys(pendingIds).length > 0

  // MANUAL: "Consistent" / "Conflict with photo"; AUTO: "Estimated dimensions
  // ..." -- from a render on screen at the current room size, none before.
  const sizeKey = JSON.stringify(size)
  const roomCheck =
    layers.map((item) => roomChecks[item.job]).find((check) => check?.size === sizeKey && check.line)
      ?.line ?? null

  // AUTO that could not size the room says so here, not only on the surface.
  const MEASURE_PROMPT = 'Please enter one known measurement'
  const roomLine =
    roomMode === 'auto' && Object.values(failures).some((reason) => reason.startsWith(MEASURE_PROMPT))
      ? MEASURE_PROMPT
      : roomCheck

  /**
   * Every note the renders on screen came with, once each — renders of
   * different surfaces report the objects and the mask clip in the same words.
   */
  const notes = [
    ...new Set([
      ...layers.flatMap((item) => jobNotes[item.job] ?? []),
      ...Object.entries(failures).map(([id, reason]) => {
        const label = found?.surfaces.find((item) => item.id === id)?.label ?? id
        return `${label} could not be tiled: ${reason}.`
      }),
    ]),
  ]

  // The white mask of the surfaces in play (selected, or still showing tiles):
  // the Clean Room job's own FLOOR_MASK.png and wall-N.png -- exactly what the
  // renderer clips to -- joined into one white-on-black image, in the browser.
  const maskSurfaces = [...new Set([...selected, ...Object.keys(shownKey)])].sort()
  const maskKey = jobId ? `${jobId}|${maskSurfaces.join(',')}` : ''

  useEffect(() => {
    if (step !== 'mask' || !jobId) return
    let cancelled = false
    const base = `${API_BASE_URL}/jobs/${jobId}/segments/`
    const urls = maskSurfaces.map((id) => (id === 'floor' ? `${base}FLOOR_MASK.png` : `${base}wall/walls/${id}.png`))
    const load = (url: string) =>
      new Promise<HTMLImageElement>((resolve, reject) => {
        const image = new Image()
        image.crossOrigin = 'anonymous'
        image.onload = () => resolve(image)
        image.onerror = () => reject(new Error(`mask unavailable: ${url}`))
        image.src = url
      })
    // No surface in play: an all-black mask, sized from the floor mask.
    const sources = urls.length ? urls : [`${base}FLOOR_MASK.png`]
    Promise.all(sources.map(load))
      .then((images) => {
        if (cancelled) return
        const canvas = document.createElement('canvas')
        canvas.width = images[0].naturalWidth
        canvas.height = images[0].naturalHeight
        const ctx = canvas.getContext('2d')!
        ctx.fillStyle = '#000'
        ctx.fillRect(0, 0, canvas.width, canvas.height)
        if (urls.length) {
          ctx.globalCompositeOperation = 'lighten'          // union: white wherever any mask is white
          for (const image of images) ctx.drawImage(image, 0, 0, canvas.width, canvas.height)
        }
        setMaskUrl(canvas.toDataURL('image/png'))
      })
      .catch((cause) => !cancelled && setError(cause instanceof Error ? cause.message : 'mask unavailable'))
    return () => {
      cancelled = true
    }
    // maskKey describes jobId + the surfaces
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [step, maskKey])

  const shown =
    step === 'before'
      ? room.previewUrl
      : step === 'mask'
        ? (maskUrl ?? room.previewUrl)
        : (composed?.result_image_url ?? room.previewUrl)

  return (
    <div className="studio">
      {/* ---------------------------------------------------------------- rail */}
      <aside className="rail">
        <div className="rail-brand">
          <Brand />
        </div>

        {blocked && (
          <p className="rail-invalid" role="alert">
            <strong>⚠ {invalid}</strong>
            <button type="button" className="btn tiny" onClick={onChangeRoom}>
              Choose another room
            </button>
          </p>
        )}

        {!cleared && !blocked && (
          <p className="rail-hint">
            {cleaning
              ? 'Finding the floor and every wall in this room…'
              : 'Preparing the room — its floor and walls will appear on the image.'}
          </p>
        )}

        {cleared && !blocked && detecting && (
          <p className="rail-hint">Finding the floor and every wall…</p>
        )}

        {cleared && !blocked && found && selected.length === 0 && (
          <p className="rail-hint">
            Tap Floor or a wall's dot to tile it with the chosen tile. Selected
            surfaces follow every tile you pick; tap one again to keep its tiles
            and pick a different tile for the others.
          </p>
        )}

        <TileRail
          tiles={catalogue?.tiles ?? []}
          selectedId={upload ? null : tile?.id}
          onSelect={(next) => {
            setUpload(null)
            setTile(next)
          }}
          onUploadClick={() => tileInput.current?.click()}
        />

        <input
          ref={tileInput}
          type="file"
          accept="image/*"
          hidden
          onChange={(event) => {
            const file = event.target.files?.[0]
            if (!file) return
            setUpload({ file, previewUrl: URL.createObjectURL(file) })
            setTile(null)
          }}
        />
      </aside>

      {/* -------------------------------------------------------------- stage */}
      <section className="stage">
        <header className="stage-top">
          <button type="button" className="icon-btn" onClick={onChangeRoom} title="Back">
            ‹
          </button>

          <button type="button" className="top-btn">💬 Get In Touch</button>
          <button type="button" className="top-btn dark">🗓 Free Trial</button>

          <div className="stage-top-right">
            <button type="button" className="top-btn" onClick={saveDesign} disabled={!composed}>
              ⊞ Add to Selection
            </button>
            <a
              className="top-btn"
              href={composed?.result_image_url ?? '#'}
              target="_blank"
              rel="noreferrer"
              aria-disabled={!composed}
            >
              ⤓ Download
            </a>
            <button type="button" className="top-btn" onClick={onChangeRoom}>
              ⌂ Change Room
            </button>
            <button type="button" className="top-btn">
              🛒 Cart <span className="badge">{designs.length}</span>
            </button>
            <button type="button" className="top-btn">Menu ⋮</button>
          </div>
        </header>

        <div className="stage-main">
          <div className="stage-canvas" ref={canvasRef}>
            <div className="step-switch" role="group" aria-label="Before, mask, after">
              {(['before', 'mask', 'after'] as const).map((item) => (
                <button
                  key={item}
                  type="button"
                  className={step === item ? 'active' : ''}
                  aria-pressed={step === item}
                  disabled={item === 'mask' && !jobId}
                  title={
                    item === 'mask'
                      ? maskSurfaces.length
                        ? `White = where tiles may go (${maskSurfaces.length} surface${maskSurfaces.length > 1 ? 's' : ''})`
                        : 'Select a surface to see its mask'
                      : item === 'before' ? 'The photograph' : 'The tiled result'
                  }
                  onClick={() => {
                    setStep(item)
                    if (item !== 'after') {
                      setCompare(false)
                      setView3d(false)
                    }
                  }}
                >
                  {item === 'before' ? 'Before' : item === 'mask' ? 'Mask' : 'After'}
                </button>
              ))}
            </div>

            {step === 'after' && compare && composed ? (
              <CompareSlider before={room.previewUrl} after={composed.result_image_url} />
            ) : (
              <img
                ref={imageRef}
                className="stage-image"
                src={shown}
                alt="Room"
                onLoad={measure}
              />
            )}

            {step === 'after' && view3d && !compare && composed && (
              <ThreeTileLayer
                layers={layers}
                frame={frame}
                onUnavailable={(reason) => {
                  setView3d(false)
                  setError(reason)
                }}
              />
            )}

            {/*
              Surfaces live on the room, not in a panel: one dot per surface
              the tile engine found. Selecting a dot tiles that surface;
              deselecting it restores the photograph there.
            */}
            {!compare && (
              <SurfaceMarkers markers={markers} onToggle={toggleSurface} frame={frame} />
            )}

            {tiling && !composed && (
              <div className="stage-busy">
                <span className="spinner" />
                <p>Laying tiles…</p>
              </div>
            )}

            {error && (
              <div className="stage-error" role="alert">
                {error}
              </div>
            )}

            <span className="stage-powered">Powered By Rich International</span>
          </div>

          {/* ------------------------------------------ the controls, beside */}
          <div className="tools">
            <button
              type="button"
              className={compare ? 'tool active' : 'tool'}
              onClick={() => {
                setStep('after')
                setCompare((value) => !value)
              }}
              disabled={!composed}
              title={composed ? 'Before / after' : 'Select a surface first'}
            >
              <span aria-hidden="true">◫</span> Compare
            </button>

            <button
              type="button"
              className={view3d ? 'tool active' : 'tool'}
              onClick={() => {
                setStep('after')
                setView3d((value) => !value)
              }}
              disabled={!composed}
              title={composed ? 'Show the tiles with the 3D renderer' : 'Select a surface first'}
            >
              <span aria-hidden="true">◈</span> 3D View
            </button>

            <button
              type="button"
              className={panel === 'grout' ? 'tool active' : 'tool'}
              onClick={() => setPanel(panel === 'grout' ? null : 'grout')}
            >
              <span aria-hidden="true">⊞</span> Grout
              <em>{grout} mm</em>
            </button>

            <button
              type="button"
              className={panel === 'layout' ? 'tool active' : 'tool'}
              onClick={() => setPanel(panel === 'layout' ? null : 'layout')}
            >
              <span aria-hidden="true">⊡</span> Layout
              <em>{rotation}°</em>
            </button>

            <button
              type="button"
              className={panel === 'size' ? 'tool active' : 'tool'}
              onClick={() => setPanel(panel === 'size' ? null : 'size')}
            >
              <span aria-hidden="true">⤢</span> Room Size
              <em>
                <span className={`mode-badge ${roomMode}`}>{roomMode.toUpperCase()}</span>
              </em>
            </button>

            <button type="button" className="tool" onClick={reset}>
              <span aria-hidden="true">⟲</span> Reset
            </button>

            {panel && (
              <div className="tool-popover">
                {panel === 'grout' && (
                  <>
                    <h4>Grout width</h4>
                    <div className="stepper">
                      <button
                        type="button"
                        onClick={() => changeGrout(grout - GROUT_STEP)}
                        disabled={grout <= GROUT_MIN}
                        aria-label="Thinner grout"
                      >
                        −
                      </button>
                      <span>{grout} mm</span>
                      <button
                        type="button"
                        onClick={() => changeGrout(grout + GROUT_STEP)}
                        disabled={grout >= GROUT_MAX}
                        aria-label="Thicker grout"
                      >
                        +
                      </button>
                    </div>
                    <input
                      className="slider"
                      type="range"
                      min={GROUT_MIN}
                      max={GROUT_MAX}
                      step={GROUT_STEP}
                      value={grout}
                      onChange={(event) => changeGrout(Number(event.target.value))}
                    />
                    <div className="popover-row">
                      {[2, 5, 8].map((value) => (
                        <button
                          key={value}
                          type="button"
                          className={grout === value ? 'pill active' : 'pill'}
                          onClick={() => changeGrout(value)}
                        >
                          {value} mm
                        </button>
                      ))}
                    </div>
                  </>
                )}

                {panel === 'layout' && (
                  <>
                    <h4>Layout rotation</h4>
                    <div className="popover-row">
                      {ROTATIONS.map((value) => (
                        <button
                          key={value}
                          type="button"
                          className={rotation === value ? 'pill active' : 'pill'}
                          onClick={() => changeRotation(value)}
                        >
                          {value}°
                        </button>
                      ))}
                    </div>
                    <h4>Tile size</h4>
                    <p className="popover-note">
                      {activeSize.width} × {activeSize.height} mm — from the selected product.
                    </p>
                  </>
                )}

                {panel === 'size' && (
                  <>
                    <h4>Room size (feet)</h4>
                    <div className="size-grid">
                      {(
                        [
                          ['height', 'Room Height'],
                          ['width', 'Room Width'],
                          ['length', 'Room Length'],
                        ] as const
                      ).map(([key, label]) => (
                        <label key={key} className="size-field">
                          <span>{label}</span>
                          <input
                            type="number"
                            min="1"
                            value={size[key]}
                            onChange={(event) =>
                              setSize((current) => ({ ...current, [key]: event.target.value }))
                            }
                          />
                        </label>
                      ))}
                    </div>
                    <p className="popover-note">
                      Used for tile scale. Applied automatically.
                    </p>
                    {roomLine && (
                      <p className="popover-note room-check" role="status">
                        {roomLine}
                      </p>
                    )}
                  </>
                )}

                {selected.length > 0 && panel !== 'size' && (
                  <p className="popover-note">Re-renders automatically after you stop adjusting.</p>
                )}
              </div>
            )}
          </div>
        </div>

        <RoomAnalysis
          room={room}
          onBusy={setCleaning}
          blocked={tiling}
          autoStart
          onCleared={(result, problem) => {
            setInvalid(problem)
            setCleared(problem ? null : result)
          }}
        />

        {/* ------------------------------------------------------- the shelves */}
        <section className="shelves">
          <article className="shelf-card">
            <header>
              <small>CURRENT ROOM</small>
              <h3>Saved Designs</h3>
              <span className="count">{designs.length}</span>
            </header>
            {designs.length === 0 ? (
              <p className="shelf-empty">Use “Add to Selection” after applying a tile.</p>
            ) : (
              <div className="shelf-grid">
                {designs.map((item) => (
                  <figure key={item.id}>
                    <img src={item.url} alt={item.label ?? item.tile} />
                    <figcaption>
                      <strong>{item.label ?? item.tile}</strong>
                      <span>{item.tile}</span>
                      <span className="shelf-surface">{item.surface}</span>
                    </figcaption>
                  </figure>
                ))}
              </div>
            )}
          </article>

          <article className="shelf-card">
            <header>
              <small>SESSION</small>
              <h3>Saved Rooms</h3>
              <span className="count">{rooms.length}</span>
            </header>
            <button type="button" className="btn tiny" onClick={saveRoom}>
              Save this room
            </button>
            {rooms.length === 0 ? (
              <p className="shelf-empty">No rooms saved yet.</p>
            ) : (
              <div className="shelf-grid">
                {rooms.map((item) => (
                  <figure key={item.id}>
                    <img src={item.url} alt={item.name} />
                    <figcaption>{item.name}</figcaption>
                  </figure>
                ))}
              </div>
            )}
          </article>

          <article className="shelf-card">
            <header>
              <small>ROOM</small>
              <h3>Surfaces</h3>
              <span className="count">{found?.surfaces.length ?? 0}</span>
            </header>
            {found?.surfaces.length ? (
              <ul className="shelf-list">
                {found.surfaces.map((item) => (
                  <li key={item.id}>
                    <span>{item.label}</span>
                    <em>{tileOn(item.id) ?? 'No tile applied'}</em>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="shelf-empty">Clean the room to find its surfaces.</p>
            )}
          </article>
        </section>
      </section>

      {/* ------------------------------------------------------------ current */}
      <aside className="nowbar">
        <div className="nowbar-tile">
          {activeThumb && <img src={activeThumb} alt="" />}
          <div>
            <small>
              {selectedLabels.length ? `Tiling ${selectedLabels.join(', ')}` : 'No surface selected'}
            </small>
            <strong>{activeName}</strong>
          </div>
        </div>

        {notes.length ? (
          <>
            <h3>Notes</h3>
            <p className="shelf-note">{notes.join(' ')}</p>
          </>
        ) : null}
      </aside>
    </div>
  )
}
