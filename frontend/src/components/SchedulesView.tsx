import { useCallback, useEffect, useState } from 'react'
import { api } from '../api/client'
import type { ModelOption, Project, Schedule, ScheduleKind } from '../types'
import { Segmented, Toggle } from './controls'
import { ClockIcon, PlayIcon, PlusIcon, TrashIcon } from './Icons'

interface Props {
  models: ModelOption[]
  projects: Project[]
  defaultModelId: string
  onOpenChat: (chatId: string) => void
  onChatsChange: () => void
}

const DAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

interface Draft {
  name: string
  prompt: string
  model_id: string
  project_id: string
  kind: ScheduleKind
  time: string
  days: number[]
  every_minutes: number
}

const emptyDraft = (modelId: string): Draft => ({
  name: '', prompt: '', model_id: modelId, project_id: '', kind: 'daily', time: '08:00', days: [0, 1, 2, 3, 4],
  every_minutes: 60,
})

function when(iso: string | null): string {
  if (!iso) return '—'
  const d = new Date(iso)
  const today = new Date()
  const tomorrow = new Date(today.getTime() + 86400000)
  const time = d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
  if (d.toDateString() === today.toDateString()) return `Today ${time}`
  if (d.toDateString() === tomorrow.toDateString()) return `Tomorrow ${time}`
  return `${d.toLocaleDateString([], { weekday: 'short', month: 'short', day: 'numeric' })} ${time}`
}

function ScheduleForm({ models, projects, initial, onSave, onCancel }: {
  models: ModelOption[]; projects: Project[]; initial: Draft; onSave: (d: Draft) => Promise<void>; onCancel: () => void
}) {
  const [d, setD] = useState(initial)
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const set = <K extends keyof Draft>(k: K, v: Draft[K]) => setD(prev => ({ ...prev, [k]: v }))

  const submit = async () => {
    setSaving(true)
    setError(null)
    try {
      await onSave(d)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not save')
    }
    setSaving(false)
  }

  return (
    <div className="schedule-form">
      <label className="field">
        <span className="field-label">Name</span>
        <input className="text-input wide" value={d.name} placeholder="Morning briefing" onChange={e => set('name', e.target.value)} />
      </label>
      <label className="field">
        <span className="field-label">What should it do?</span>
        <textarea className="text-area" rows={4} value={d.prompt} onChange={e => set('prompt', e.target.value)}
                  placeholder="Summarize today's top tech news and the weather in Chicago." />
      </label>
      <div className="field">
        <span className="field-label">When</span>
        <Segmented value={d.kind} onChange={v => set('kind', v)} options={[
          { value: 'daily', label: 'Daily' }, { value: 'weekly', label: 'Weekly' }, { value: 'interval', label: 'Repeating' },
        ]} />
        <div className="field-row">
          {d.kind !== 'interval' ? (<>
            <input type="time" className="text-input narrow" value={d.time} onChange={e => set('time', e.target.value)} aria-label="Time" />
            {d.kind === 'weekly' && (
              <div className="day-picker" role="group" aria-label="Days">
                {DAY_NAMES.map((n, i) => (
                  <button key={n} className={d.days.includes(i) ? 'on' : ''} aria-pressed={d.days.includes(i)}
                          onClick={() => set('days', d.days.includes(i) ? d.days.filter(x => x !== i) : [...d.days, i].sort())}>
                    {n}
                  </button>
                ))}
              </div>
            )}
          </>) : (
            <label className="inline-label">Every
              <select className="select narrow" value={d.every_minutes} onChange={e => set('every_minutes', Number(e.target.value))}>
                {[15, 30, 60, 120, 240, 360, 720].map(m => <option key={m} value={m}>{m < 60 ? `${m} minutes` : `${m / 60} hour${m === 60 ? '' : 's'}`}</option>)}
              </select>
            </label>
          )}
        </div>
      </div>
      <div className="field-row">
        <label className="field grow">
          <span className="field-label">Model</span>
          <select className="select" value={d.model_id} onChange={e => set('model_id', e.target.value)}>
            {models.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}
          </select>
        </label>
        <label className="field grow">
          <span className="field-label">Project</span>
          <select className="select" value={d.project_id} onChange={e => set('project_id', e.target.value)}>
            <option value="">None</option>
            {projects.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}
          </select>
        </label>
      </div>
      <p className="setting-hint">
        Runs in the background while the backend is running; each run is posted to this schedule's own chat. Nobody is
        there to answer, so questions are skipped and actions that need approval are declined.
      </p>
      {error && <p className="form-error">{error}</p>}
      <div className="form-actions">
        <button className="btn" onClick={onCancel}>Cancel</button>
        <button className="btn primary" onClick={submit} disabled={saving || !d.prompt.trim() || (d.kind === 'weekly' && !d.days.length)}>
          {saving ? 'Saving…' : 'Save'}
        </button>
      </div>
    </div>
  )
}

export default function SchedulesView({ models, projects, defaultModelId, onOpenChat, onChatsChange }: Props) {
  const [items, setItems] = useState<Schedule[] | null>(null)
  const [editing, setEditing] = useState<string | 'new' | null>(null)

  const load = useCallback(() => api.listSchedules().then(setItems), [])
  useEffect(() => {
    load()
    const id = setInterval(load, 30000)
    return () => clearInterval(id)
  }, [load])

  const save = (id: string | 'new') => async (d: Draft) => {
    const body: Partial<Schedule> = {
      name: d.name, prompt: d.prompt, model_id: d.model_id || null, project_id: d.project_id || null, kind: d.kind,
      ...(d.kind === 'interval' ? { every_minutes: d.every_minutes } : { time: d.time }),
      ...(d.kind === 'weekly' ? { days: d.days } : {}),
    }
    if (id === 'new') await api.createSchedule(body)
    else await api.updateSchedule(id, body)
    setEditing(null)
    load()
  }

  const toDraft = (s: Schedule): Draft => ({
    name: s.name, prompt: s.prompt, model_id: s.model_id ?? defaultModelId, project_id: s.project_id ?? '', kind: s.kind,
    time: s.time ?? '08:00', days: s.days ?? [0, 1, 2, 3, 4], every_minutes: s.every_minutes ?? 60,
  })

  return (
    <div className="page">
      <div className="page-scroll">
        <div className="page-inner narrow">
          <div className="page-head">
            <h1 className="settings-title">Scheduled</h1>
            {editing === null && (
              <button className="btn primary" onClick={() => setEditing('new')}><PlusIcon size={14} /> New schedule</button>
            )}
          </div>
          <p className="settings-intro">Prompts that run on their own — a morning briefing, a weekly report, a check every few hours.
            Each one gets a chat where its runs appear.</p>

          {editing === 'new' && (
            <ScheduleForm models={models} projects={projects} initial={emptyDraft(defaultModelId)}
                          onSave={save('new')} onCancel={() => setEditing(null)} />
          )}

          {items === null && <p className="muted-text">Loading…</p>}
          {items?.length === 0 && editing !== 'new' && (
            <div className="empty-state">
              <ClockIcon size={28} />
              <p>No schedules yet.</p>
            </div>
          )}

          {items?.map(s => editing === s.id ? (
            <ScheduleForm key={s.id} models={models} projects={projects} initial={toDraft(s)}
                          onSave={save(s.id)} onCancel={() => setEditing(null)} />
          ) : (
            <div key={s.id} className={`schedule-card${s.enabled ? '' : ' paused'}`}>
              <div className="schedule-main" onClick={() => setEditing(s.id)} role="button" tabIndex={0}
                   onKeyDown={e => { if (e.key === 'Enter') setEditing(s.id) }}>
                <div className="schedule-name">{s.name}</div>
                <div className="schedule-when">{s.description}{s.enabled ? ` · next ${when(s.next_run_at)}` : ' · paused'}</div>
                <div className="schedule-prompt">{s.prompt}</div>
                {s.last_run_at && <div className="schedule-last">Last ran {when(s.last_run_at)}</div>}
              </div>
              <div className="schedule-actions">
                <Toggle checked={s.enabled} onChange={async v => { await api.updateSchedule(s.id, { enabled: v }); load() }} />
                <button className="icon-btn small" title="Run now" aria-label="Run now" onClick={async () => {
                  const { chat_id } = await api.runSchedule(s.id)
                  onChatsChange()
                  onOpenChat(chat_id)
                }}><PlayIcon size={15} /></button>
                {s.chat_id && <button className="btn small" onClick={() => onOpenChat(s.chat_id!)}>Open chat</button>}
                <button className="icon-btn small" title="Delete" aria-label="Delete schedule" onClick={async () => {
                  if (!confirm(`Delete "${s.name}"? Its chat is kept.`)) return
                  await api.deleteSchedule(s.id)
                  load()
                }}><TrashIcon size={15} /></button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
