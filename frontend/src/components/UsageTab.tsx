import { useEffect, useState } from 'react'
import { api } from '../api/client'
import type { UsageReport } from '../types'

interface Props {
  budget: number | null
  onBudget: (v: number | null) => void
}

const RANGES = [7, 30, 90]

function usd(v: number): string {
  if (v === 0) return '$0'
  if (v < 0.01) return '<$0.01'
  return v < 100 ? `$${v.toFixed(2)}` : `$${Math.round(v).toLocaleString()}`
}

function tokens(n: number): string {
  if (n < 1000) return String(n)
  if (n < 1_000_000) return `${(n / 1000).toFixed(n < 10_000 ? 1 : 0)}K`
  return `${(n / 1_000_000).toFixed(1)}M`
}

/** Axis label with enough decimals for the scale (tiny local-ish spend still gets readable ticks). */
function axisUsd(v: number, max: number): string {
  if (v === 0) return '$0'
  const decimals = max >= 10 ? 0 : max >= 0.1 ? 2 : Math.min(6, Math.ceil(-Math.log10(max)) + 1)
  return `$${v.toFixed(decimals)}`
}

/** A "nice" axis maximum ≥ v (1, 2, 2.5, 5 × 10^n). */
function niceMax(v: number): number {
  if (v <= 0) return 1
  const p = 10 ** Math.floor(Math.log10(v))
  for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= v) return m * p
  return 10 * p
}

const shortDate = (d: string) => new Date(`${d}T00:00`).toLocaleDateString([], { month: 'short', day: 'numeric' })

/** Daily spend: one series, so no legend — the heading names it. Hover a bar for its numbers. */
function DailyChart({ days }: { days: UsageReport['days'] }) {
  const [hover, setHover] = useState<number | null>(null)
  const W = 640, H = 180, L = 56, B = 22, T = 8
  const max = niceMax(Math.max(...days.map(d => d.cost_usd)))
  const step = (W - L) / days.length
  const bar = Math.max(2, step - 2)          // 2px gap between bars
  const y = (v: number) => T + (H - T - B) * (1 - v / max)
  const ticks = [0, max / 2, max]
  const labelEvery = Math.ceil(days.length / 6)
  const h = hover !== null ? days[hover] : null

  return (
    <div className="usage-chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Spend per day" onMouseLeave={() => setHover(null)}>
        {ticks.map(t => (
          <g key={t}>
            <line x1={L} x2={W} y1={y(t)} y2={y(t)} className="usage-grid" />
            <text x={L - 6} y={y(t)} className="usage-axis" textAnchor="end" dominantBaseline="middle">{axisUsd(t, max)}</text>
          </g>
        ))}
        {days.map((d, i) => {
          const x = L + i * step + 1
          const top = y(d.cost_usd)
          const h = H - B - top
          const r = Math.min(4, bar / 2, h)
          return (
            <g key={d.date} onMouseEnter={() => setHover(i)}>
              {/* hit target: the whole column, bigger than the mark */}
              <rect x={L + i * step} y={T} width={step} height={H - T - B} fill="transparent" />
              {d.cost_usd > 0 && (
                <path className={`usage-bar${hover === i ? ' hover' : ''}`}
                      d={`M${x},${H - B} V${top + r} Q${x},${top} ${x + r},${top} H${x + bar - r} Q${x + bar},${top} ${x + bar},${top + r} V${H - B} Z`} />
              )}
              {i % labelEvery === 0 && (
                <text x={x + bar / 2} y={H - 6} className="usage-axis" textAnchor="middle">{shortDate(d.date)}</text>
              )}
            </g>
          )
        })}
        <line x1={L} x2={W} y1={H - B} y2={H - B} className="usage-baseline" />
      </svg>
      {h && hover !== null && (
        <div className="usage-tooltip" style={{ left: `${((L + hover * step + step / 2) / W) * 100}%` }}>
          <strong>{shortDate(h.date)}</strong>
          <span>{usd(h.cost_usd)} · {h.turns} turn{h.turns === 1 ? '' : 's'}</span>
          <span className="muted-text">{tokens(h.input_tokens)} in · {tokens(h.output_tokens)} out</span>
        </div>
      )}
    </div>
  )
}

export default function UsageTab({ budget, onBudget }: Props) {
  const [days, setDays] = useState(30)
  const [report, setReport] = useState<UsageReport | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.usage(days).then(setReport).catch(e => setError(e.message))
  }, [days])

  if (error) return <p className="form-error">{error}</p>
  if (!report) return <p className="muted-text">Loading…</p>

  const month = report.month_to_date_usd
  const pct = budget ? Math.min(100, (month / budget) * 100) : 0
  const over = budget !== null && month >= budget

  return (<>
    <p className="settings-intro">
      What API calls have cost, from the token counts each provider reports. Local models are free and show tokens only.
    </p>

    <div className="stat-tiles">
      <div className="stat-tile">
        <span className="stat-label">This month</span>
        <span className="stat-value">{usd(month)}</span>
        {budget ? (<>
          <div className="meter" role="meter" aria-valuemin={0} aria-valuemax={budget} aria-valuenow={month}
               aria-label="Spend against monthly limit">
            <span style={{ width: `${pct}%` }} className={over ? 'over' : ''} />
          </div>
          <span className="stat-sub">{over ? 'Limit reached — new messages are paused' : `of ${usd(budget)} limit`}</span>
        </>) : <span className="stat-sub">No monthly limit</span>}
      </div>
      <div className="stat-tile">
        <span className="stat-label">Last {days} days</span>
        <span className="stat-value">{usd(report.total.cost_usd)}</span>
        <span className="stat-sub">{report.total.turns} turns</span>
      </div>
      <div className="stat-tile">
        <span className="stat-label">Tokens ({days} days)</span>
        <span className="stat-value">{tokens(report.total.input_tokens + report.total.output_tokens)}</span>
        <span className="stat-sub">{tokens(report.total.input_tokens)} in · {tokens(report.total.output_tokens)} out</span>
      </div>
    </div>

    <div className="usage-head">
      <h2 className="settings-subhead">Spend per day</h2>
      <div className="segmented small" role="radiogroup">
        {RANGES.map(r => (
          <button key={r} role="radio" aria-checked={days === r} className={days === r ? 'on' : ''} onClick={() => setDays(r)}>
            {r}d
          </button>
        ))}
      </div>
    </div>
    <DailyChart days={report.days} />
    <details className="usage-table-toggle">
      <summary>Show as table</summary>
      <table className="usage-table">
        <thead><tr><th>Day</th><th>Turns</th><th>Input</th><th>Output</th><th>Cost</th></tr></thead>
        <tbody>
          {report.days.filter(d => d.turns).reverse().map(d => (
            <tr key={d.date}><td>{shortDate(d.date)}</td><td>{d.turns}</td><td>{tokens(d.input_tokens)}</td>
              <td>{tokens(d.output_tokens)}</td><td>{usd(d.cost_usd)}</td></tr>
          ))}
        </tbody>
      </table>
    </details>

    <h2 className="settings-subhead">By model</h2>
    {report.models.length === 0 ? <p className="muted-text">No usage in this period.</p> : (
      <table className="usage-table">
        <thead><tr><th>Model</th><th>Calls</th><th>Input</th><th>Output</th><th>Cost</th></tr></thead>
        <tbody>
          {report.models.map(m => (
            <tr key={m.model_id}><td>{m.name}</td><td>{m.calls}</td><td>{tokens(m.input_tokens)}</td>
              <td>{tokens(m.output_tokens)}</td><td>{usd(m.cost_usd)}</td></tr>
          ))}
        </tbody>
      </table>
    )}

    {report.chats.some(c => c.cost_usd > 0) && (<>
      <h2 className="settings-subhead">Most expensive chats</h2>
      <table className="usage-table">
        <thead><tr><th>Chat</th><th>Turns</th><th>Cost</th></tr></thead>
        <tbody>
          {report.chats.filter(c => c.cost_usd > 0).map(c => (
            <tr key={c.chat_id ?? 'deleted'}><td>{c.name}</td><td>{c.turns}</td><td>{usd(c.cost_usd)}</td></tr>
          ))}
        </tbody>
      </table>
    </>)}

    <h2 className="settings-subhead">Limit</h2>
    <div className="setting-row">
      <div className="setting-label">
        <span className="setting-name">Monthly spending limit</span>
        <span className="setting-hint">New messages and scheduled runs are refused once this calendar month's spend reaches it.</span>
      </div>
      <div className="setting-control">
        <select className="select" value={String(budget ?? 0)} onChange={e => onBudget(Number(e.target.value) || null)}>
          <option value="0">No limit</option>
          {[1, 5, 10, 20, 50, 100, 250, 500].map(n => <option key={n} value={String(n)}>${n} / month</option>)}
        </select>
      </div>
    </div>
  </>)
}
