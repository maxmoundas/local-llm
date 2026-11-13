import type { ReactNode } from 'react'

export function Toggle({ checked, onChange, disabled }: { checked: boolean; onChange: (v: boolean) => void; disabled?: boolean }) {
  return (
    <label className={`toggle${disabled ? ' disabled' : ''}`}>
      <input type="checkbox" checked={checked} disabled={disabled} onChange={e => onChange(e.target.checked)} />
      <span className="toggle-track"><span className="toggle-thumb" /></span>
    </label>
  )
}

export function Row({ name, hint, children, stacked }: { name: string; hint?: ReactNode; children: ReactNode; stacked?: boolean }) {
  return (
    <div className={`setting-row${stacked ? ' stacked' : ''}`}>
      <div className="setting-label">
        <span className="setting-name">{name}</span>
        {hint && <span className="setting-hint">{hint}</span>}
      </div>
      <div className="setting-control">{children}</div>
    </div>
  )
}

export function Segmented<T extends string>({ value, options, onChange }: {
  value: T; options: { value: T; label: string }[]; onChange: (v: T) => void
}) {
  return (
    <div className="segmented" role="radiogroup">
      {options.map(o => (
        <button key={o.value} role="radio" aria-checked={value === o.value}
                className={value === o.value ? 'on' : ''} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  )
}
