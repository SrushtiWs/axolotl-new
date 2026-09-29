import { CUSTOM_TILE_ID, TILE_PRESETS } from '../types'

interface TileSizeSelectorProps {
  presetId: string
  custom: { width: string; height: string }
  error?: string
  onPresetChange: (id: string) => void
  onCustomChange: (next: { width: string; height: string }) => void
}

export function TileSizeSelector({
  presetId,
  custom,
  error,
  onPresetChange,
  onCustomChange,
}: TileSizeSelectorProps) {
  const isCustom = presetId === CUSTOM_TILE_ID

  return (
    <div className="tile-size">
      <div className="chip-row">
        {TILE_PRESETS.map((preset) => (
          <button
            key={preset.id}
            type="button"
            className={`chip ${presetId === preset.id ? 'selected' : ''}`}
            aria-pressed={presetId === preset.id}
            onClick={() => onPresetChange(preset.id)}
          >
            {preset.label}
          </button>
        ))}
        <button
          type="button"
          className={`chip ${isCustom ? 'selected' : ''}`}
          aria-pressed={isCustom}
          onClick={() => onPresetChange(CUSTOM_TILE_ID)}
        >
          Custom
        </button>
      </div>

      {isCustom && (
        <div className="custom-tile">
          <label className="dimension-field">
            <span className="field-label">Tile Width</span>
            <span className="input-wrap">
              <input
                type="number"
                min="1"
                step="1"
                inputMode="numeric"
                placeholder="0"
                className={error ? 'invalid' : ''}
                value={custom.width}
                onChange={(event) =>
                  onCustomChange({ ...custom, width: event.target.value })
                }
              />
              <span className="unit">mm</span>
            </span>
          </label>
          <label className="dimension-field">
            <span className="field-label">Tile Height</span>
            <span className="input-wrap">
              <input
                type="number"
                min="1"
                step="1"
                inputMode="numeric"
                placeholder="0"
                className={error ? 'invalid' : ''}
                value={custom.height}
                onChange={(event) =>
                  onCustomChange({ ...custom, height: event.target.value })
                }
              />
              <span className="unit">mm</span>
            </span>
          </label>
        </div>
      )}

      {error && <p className="field-error">{error}</p>}
    </div>
  )
}
