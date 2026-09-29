import type { RoomDimensions as Dimensions, ValidationErrors } from '../types'

interface RoomDimensionsProps {
  value: Dimensions
  errors: ValidationErrors
  onChange: (next: Dimensions) => void
}

const FIELDS = [
  { key: 'width', label: 'Width', errorKey: 'roomWidth' },
  { key: 'length', label: 'Length', errorKey: 'roomLength' },
  { key: 'height', label: 'Height', errorKey: 'roomHeight' },
] as const

export function RoomDimensions({ value, errors, onChange }: RoomDimensionsProps) {
  return (
    <div className="dimension-grid">
      {FIELDS.map(({ key, label, errorKey }) => {
        const error = errors[errorKey]

        return (
          <label key={key} className="dimension-field">
            <span className="field-label">{label}</span>
            <span className="input-wrap">
              <input
                type="number"
                min="0"
                step="0.1"
                inputMode="decimal"
                placeholder="0"
                className={error ? 'invalid' : ''}
                value={value[key]}
                onChange={(event) => onChange({ ...value, [key]: event.target.value })}
              />
              <span className="unit">ft</span>
            </span>
            {error && <span className="field-error">{error}</span>}
          </label>
        )
      })}
    </div>
  )
}
