interface Option<T extends string | number> {
  value: T
  label: string
}

interface OptionGroupProps<T extends string | number> {
  name: string
  options: Option<T>[]
  value: T
  onChange: (value: T) => void
}

/** Radio group used for both tile rotation and surface selection. */
export function OptionGroup<T extends string | number>({
  name,
  options,
  value,
  onChange,
}: OptionGroupProps<T>) {
  return (
    <div className="radio-row" role="radiogroup" aria-label={name}>
      {options.map((option) => (
        <label
          key={String(option.value)}
          className={`radio-item ${value === option.value ? 'selected' : ''}`}
        >
          <input
            type="radio"
            name={name}
            checked={value === option.value}
            onChange={() => onChange(option.value)}
          />
          <span className="radio-dot" aria-hidden="true" />
          <span>{option.label}</span>
        </label>
      ))}
    </div>
  )
}
