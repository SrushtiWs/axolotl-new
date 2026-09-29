import { useCallback, useEffect, useRef, useState } from 'react'
import { ImageDropzone } from '../components/ImageDropzone'
import { OptionGroup } from '../components/OptionGroup'
import { PipelineFlow } from '../components/PipelineFlow'
import { ProgressPanel } from '../components/ProgressPanel'
import { ResultPanel } from '../components/ResultPanel'
import { RoomDimensions } from '../components/RoomDimensions'
import { SegmentationPanel } from '../components/SegmentationPanel'
import { Section } from '../components/Section'
import { TileSizeSelector } from '../components/TileSizeSelector'
import { generateVisualization, isBackendConfigured } from '../services/api'
import {
  CUSTOM_TILE_ID,
  PIPELINE_STEPS,
  ROTATIONS,
  SURFACES,
  TILE_PRESETS,
} from '../types'
import type {
  GenerateResponse,
  GenerationStatus,
  PipelineStep,
  Rotation,
  Surface,
  TileSize,
  UploadedImage,
  ValidationErrors,
  VisualizerForm,
} from '../types'

const INITIAL_FORM: VisualizerForm = {
  roomImage: null,
  tileImage: null,
  room: { width: '', length: '', height: '' },
  tilePresetId: '',
  customTile: { width: '', height: '' },
  rotation: 0,
  surface: 'floor',
}

/** Milliseconds each simulated progress step is shown for. */
const STEP_DELAY = 700

function resolveTileSize(form: VisualizerForm): TileSize | null {
  if (form.tilePresetId === CUSTOM_TILE_ID) {
    const width = Number(form.customTile.width)
    const height = Number(form.customTile.height)

    if (!Number.isFinite(width) || !Number.isFinite(height)) return null
    if (width <= 0 || height <= 0) return null

    return { width, height }
  }

  const preset = TILE_PRESETS.find((item) => item.id === form.tilePresetId)
  return preset ? { width: preset.width, height: preset.height } : null
}

function validate(form: VisualizerForm): ValidationErrors {
  const errors: ValidationErrors = {}

  if (!form.roomImage) errors.roomImage = 'Room image is required.'
  if (!form.tileImage) errors.tileImage = 'Tile image is required.'

  const dimensions = [
    ['width', 'roomWidth', 'Width'],
    ['length', 'roomLength', 'Length'],
    ['height', 'roomHeight', 'Height'],
  ] as const

  for (const [key, errorKey, label] of dimensions) {
    const raw = form.room[key].trim()

    if (!raw) {
      errors[errorKey] = `${label} is required.`
      continue
    }

    const value = Number(raw)

    if (!Number.isFinite(value) || value <= 0) {
      errors[errorKey] = `${label} must be greater than 0.`
    }
  }

  if (!form.tilePresetId) {
    errors.tileSize = 'Tile size is required.'
  } else if (!resolveTileSize(form)) {
    errors.tileSize = 'Enter a valid custom tile width and height in mm.'
  }

  if (!ROTATIONS.includes(form.rotation)) errors.rotation = 'Rotation is required.'
  if (!form.surface) errors.surface = 'Surface is required.'

  return errors
}

export function TileVisualizer() {
  const [form, setForm] = useState<VisualizerForm>(INITIAL_FORM)
  const [errors, setErrors] = useState<ValidationErrors>({})
  const [submitted, setSubmitted] = useState(false)
  const [status, setStatus] = useState<GenerationStatus>('idle')
  const [steps, setSteps] = useState<PipelineStep[]>([])
  const [result, setResult] = useState<GenerateResponse | null>(null)
  const [message, setMessage] = useState<string | null>(null)

  const timers = useRef<number[]>([])
  const objectUrls = useRef<string[]>([])

  const clearTimers = useCallback(() => {
    timers.current.forEach((id) => window.clearTimeout(id))
    timers.current = []
  }, [])

  useEffect(() => {
    return () => {
      clearTimers()
      objectUrls.current.forEach((url) => URL.revokeObjectURL(url))
    }
  }, [clearTimers])

  function track(file: File): UploadedImage {
    const previewUrl = URL.createObjectURL(file)
    objectUrls.current.push(previewUrl)
    return { file, previewUrl }
  }

  function release(image: UploadedImage | null) {
    if (!image) return
    URL.revokeObjectURL(image.previewUrl)
    objectUrls.current = objectUrls.current.filter((url) => url !== image.previewUrl)
  }

  function update(next: Partial<VisualizerForm>) {
    setForm((current) => {
      const merged = { ...current, ...next }
      if (submitted) setErrors(validate(merged))
      return merged
    })
  }

  function setRoomImage(file: File | null) {
    release(form.roomImage)
    update({ roomImage: file ? track(file) : null })
  }

  function setTileImage(file: File | null) {
    release(form.tileImage)
    update({ tileImage: file ? track(file) : null })
  }

  function runProgressSimulation(onComplete: () => void) {
    const initial: PipelineStep[] = PIPELINE_STEPS.map((step, index) => ({
      ...step,
      status: index === 0 ? 'active' : 'pending',
    }))

    setSteps(initial)

    PIPELINE_STEPS.forEach((_step, index) => {
      const id = window.setTimeout(
        () => {
          setSteps((current) =>
            current.map((step, position) => {
              if (position <= index) return { ...step, status: 'done' }
              if (position === index + 1) return { ...step, status: 'active' }
              return step
            }),
          )

          if (index === PIPELINE_STEPS.length - 1) onComplete()
        },
        STEP_DELAY * (index + 1),
      )

      timers.current.push(id)
    })
  }

  async function handleGenerate() {
    setSubmitted(true)

    const found = validate(form)
    setErrors(found)

    if (Object.keys(found).length > 0) {
      setStatus('idle')
      setMessage(null)
      return
    }

    const tileSize = resolveTileSize(form)
    if (!tileSize || !form.roomImage || !form.tileImage) return

    clearTimers()
    setResult(null)
    setMessage(null)
    setStatus('running')

    const request = {
      room_image: form.roomImage.file,
      tile_image: form.tileImage.file,
      room_width: Number(form.room.width),
      room_length: Number(form.room.length),
      room_height: Number(form.room.height),
      tile_width: tileSize.width,
      tile_height: tileSize.height,
      rotation: form.rotation,
      surface: form.surface,
    }

    runProgressSimulation(async () => {
      // No backend is wired up yet, so nothing is fabricated here — the UI
      // reports that the pipeline is not connected instead of showing a
      // pretend result. Once VITE_API_BASE_URL is set, the real call runs.
      if (!isBackendConfigured()) {
        setStatus('not_connected')
        setMessage(
          'Frontend is ready. No backend is connected yet, so no image was generated. ' +
            'Set VITE_API_BASE_URL to the server that implements POST /generate.',
        )
        return
      }

      try {
        const response = await generateVisualization(request)
        setResult(response)
        // The server says when a camera was estimated rather than measured;
        // that caveat belongs next to the image, not buried in a JSON field.
        setMessage(response.notes?.join(' ') ?? null)
        setStatus('done')
      } catch (error) {
        setStatus('error')
        setMessage(error instanceof Error ? error.message : 'Generation failed.')
      }
    })
  }

  function handleReset() {
    clearTimers()
    release(form.roomImage)
    release(form.tileImage)
    setForm(INITIAL_FORM)
    setErrors({})
    setSubmitted(false)
    setStatus('idle')
    setSteps([])
    setResult(null)
    setMessage(null)
  }

  const errorList = Object.values(errors)
  const isRunning = status === 'running'

  return (
    <div className="page">
      <header className="app-header">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true" />
          <div>
            <h1>AI Tile Visualizer</h1>
            <p>See your tile in a real room before you buy.</p>
          </div>
        </div>
        <button type="button" className="btn subtle" onClick={handleReset}>
          Reset
        </button>
      </header>

      <main className="layout">
        <div className="column form-column">
          <Section title="Room Image" required>
            <ImageDropzone
              label="Room image"
              dropHint="Drag &amp; Drop image here"
              buttonLabel="Choose Room Image"
              replaceLabel="Replace image"
              variant="room"
              image={form.roomImage}
              error={errors.roomImage}
              onSelect={(file) => setRoomImage(file)}
              onClear={() => setRoomImage(null)}
            />
          </Section>

          <Section
            title="Tile Image"
            required
            hint="Your real tile artwork. It is sent to the backend unchanged."
          >
            <ImageDropzone
              label="Tile image"
              dropHint="Drag &amp; Drop tile image here"
              buttonLabel="Choose Tile Image"
              replaceLabel="Replace tile"
              variant="tile"
              image={form.tileImage}
              error={errors.tileImage}
              onSelect={(file) => setTileImage(file)}
              onClear={() => setTileImage(null)}
            />
          </Section>

          <Section title="Room Dimensions" required>
            <RoomDimensions
              value={form.room}
              errors={errors}
              onChange={(room) => update({ room })}
            />
          </Section>

          <Section title="Tile Size" required>
            <TileSizeSelector
              presetId={form.tilePresetId}
              custom={form.customTile}
              error={errors.tileSize}
              onPresetChange={(tilePresetId) => update({ tilePresetId })}
              onCustomChange={(customTile) => update({ customTile })}
            />
          </Section>

          <Section title="Tile Rotation" required>
            <OptionGroup<Rotation>
              name="rotation"
              value={form.rotation}
              options={ROTATIONS.map((value) => ({ value, label: `${value}°` }))}
              onChange={(rotation) => update({ rotation })}
            />
          </Section>

          <Section title="Apply Tile To" required>
            <OptionGroup<Surface>
              name="surface"
              value={form.surface}
              options={SURFACES}
              onChange={(surface) => update({ surface })}
            />
          </Section>

          {submitted && errorList.length > 0 && (
            <div className="notice error" role="alert">
              <strong>Please complete the following:</strong>
              <ul>
                {errorList.map((error) => (
                  <li key={error}>{error}</li>
                ))}
              </ul>
            </div>
          )}

          <button
            type="button"
            className="btn generate"
            onClick={handleGenerate}
            disabled={isRunning}
          >
            {isRunning ? 'Generating...' : 'GENERATE'}
          </button>
        </div>

        <div className="column result-column">
          <Section title="Result">
            {isRunning ? (
              <ProgressPanel steps={steps} />
            ) : (
              <ResultPanel
                status={status}
                roomImage={form.roomImage}
                result={result}
                message={message}
              />
            )}
          </Section>

          <SegmentationPanel roomImage={form.roomImage} />

          <PipelineFlow />
        </div>
      </main>
    </div>
  )
}
