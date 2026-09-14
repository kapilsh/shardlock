// Small controlled form primitives used by the sidebar panels.

export function Section({ title, children, aside }) {
  return (
    <section className="side-section">
      <div className="side-section-head">
        <h3>{title}</h3>
        {aside}
      </div>
      {children}
    </section>
  )
}

export function NumberField({ label, value, onChange, min = 0, step = 1, hint, disabled }) {
  return (
    <label className={`field ${disabled ? 'disabled' : ''}`} title={hint}>
      <span className="field-label">{label}</span>
      <input
        type="number"
        className="mono num"
        value={value}
        min={min}
        step={step}
        disabled={disabled}
        onChange={(e) => {
          const v = e.target.value === '' ? min : Number(e.target.value)
          if (Number.isFinite(v)) onChange(v)
        }}
      />
    </label>
  )
}

export function SelectField({ label, value, onChange, options, disabled, hint }) {
  return (
    <label className={`field ${disabled ? 'disabled' : ''}`} title={hint}>
      <span className="field-label">{label}</span>
      <select value={value} disabled={disabled} onChange={(e) => onChange(e.target.value)}>
        {Object.entries(options).map(([k, v]) => (
          <option key={k} value={k}>
            {v}
          </option>
        ))}
      </select>
    </label>
  )
}

export function Toggle({ label, checked, onChange, disabled, hint }) {
  return (
    <label className={`toggle ${disabled ? 'disabled' : ''}`} title={hint}>
      <input type="checkbox" checked={checked} disabled={disabled} onChange={(e) => onChange(e.target.checked)} />
      <span>{label}</span>
    </label>
  )
}

export function Segmented({ value, onChange, options }) {
  return (
    <div className="segmented" role="radiogroup">
      {Object.entries(options).map(([k, v]) => (
        <button
          key={k}
          type="button"
          role="radio"
          aria-checked={value === k}
          className={value === k ? 'on' : ''}
          onClick={() => onChange(k)}
        >
          {v}
        </button>
      ))}
    </div>
  )
}

// Power-of-two stepper for parallel degrees.
export function DegreeField({ label, value, onChange, color, max = 1024, hint }) {
  return (
    <div className="degree" title={hint}>
      <span className="degree-label">
        <span className="swatch" style={{ background: color }} />
        {label}
      </span>
      <div className="degree-ctl">
        <button type="button" aria-label={`halve ${label}`} disabled={value <= 1} onClick={() => onChange(Math.max(1, Math.floor(value / 2)))}>
          ÷2
        </button>
        <input
          type="number"
          className="mono num"
          min={1}
          value={value}
          onChange={(e) => {
            const v = Math.floor(Number(e.target.value))
            if (v >= 1) onChange(Math.min(v, max))
          }}
        />
        <button type="button" aria-label={`double ${label}`} disabled={value >= max} onClick={() => onChange(Math.min(max, value * 2))}>
          ×2
        </button>
      </div>
    </div>
  )
}
