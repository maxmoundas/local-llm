import { useEffect, useRef, useState } from 'react'
import { api } from '../api/client'
import type { ChatSettings, ModelOption } from '../types'
import { Segmented } from './controls'
import { SlidersIcon } from './Icons'

/** Per-chat model settings: a custom system prompt, temperature, and reasoning effort. */
export default function ChatSettingsMenu({ chatId, model }: { chatId: string; model?: ModelOption }) {
  const [open, setOpen] = useState(false)
  const [saved, setSaved] = useState<ChatSettings>({})
  const [draft, setDraft] = useState<ChatSettings>({})
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    api.getChat(chatId).then(c => { setSaved(c.settings ?? {}); setDraft(c.settings ?? {}) }).catch(() => {})
  }, [chatId])

  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => { if (!ref.current?.contains(e.target as Node)) setOpen(false) }
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false) }
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey)
    return () => { document.removeEventListener('mousedown', onDown); document.removeEventListener('keydown', onKey) }
  }, [open])

  const custom = !!(saved.system_prompt || saved.temperature !== undefined || saved.effort)
  const dirty = JSON.stringify(draft) !== JSON.stringify(saved)
  const save = async () => {
    const clean: ChatSettings = {}
    if (draft.system_prompt?.trim()) clean.system_prompt = draft.system_prompt.trim()
    if (draft.temperature !== undefined) clean.temperature = draft.temperature
    if (draft.effort) clean.effort = draft.effort
    const chat = await api.updateChatSettings(chatId, clean)
    setSaved(chat.settings ?? {})
    setDraft(chat.settings ?? {})
    setOpen(false)
  }

  const tempSupported = model?.temperature !== false
  const effortSupported = !!model?.effort

  return (
    <div className="chat-settings" ref={ref}>
      <button className={`icon-btn${custom ? ' accent' : ''}`} onClick={() => setOpen(o => !o)}
              title="Chat settings" aria-label="Chat settings" aria-expanded={open}>
        <SlidersIcon size={17} />
      </button>
      {open && (
        <div className="popover chat-settings-pop">
          <h3>This chat</h3>
          <label className="field">
            <span className="field-label">Instructions</span>
            <textarea className="text-area" rows={4} value={draft.system_prompt ?? ''}
                      placeholder="Replaces your global system prompt for this chat only."
                      onChange={e => setDraft({ ...draft, system_prompt: e.target.value })} />
          </label>

          <div className="field">
            <span className="field-label">
              Temperature {draft.temperature !== undefined ? <strong>{draft.temperature.toFixed(1)}</strong>
                : tempSupported ? <span className="muted-text">(model default)</span> : null}
            </span>
            {tempSupported ? (
              <div className="field-row">
                <input type="range" min={0} max={2} step={0.1} value={draft.temperature ?? 1}
                       onChange={e => setDraft({ ...draft, temperature: Number(e.target.value) })} aria-label="Temperature" />
                {draft.temperature !== undefined && (
                  <button className="link-btn" onClick={() => setDraft({ ...draft, temperature: undefined })}>Reset</button>
                )}
              </div>
            ) : <span className="setting-hint">{model?.name ?? 'This model'} doesn't accept a temperature.</span>}
          </div>

          <div className="field">
            <span className="field-label">Reasoning effort</span>
            {effortSupported ? (
              <Segmented<'default' | 'low' | 'medium' | 'high'> value={draft.effort ?? 'default'} onChange={v => setDraft({ ...draft, effort: v === 'default' ? undefined : v })}
                         options={[{ value: 'default', label: 'Default' }, { value: 'low', label: 'Low' },
                                   { value: 'medium', label: 'Medium' }, { value: 'high', label: 'High' }]} />
            ) : <span className="setting-hint">{model?.name ?? 'This model'} has no effort setting.</span>}
            {effortSupported && <span className="setting-hint">Higher thinks longer before answering: better on hard problems, slower and pricier.</span>}
          </div>

          <div className="form-actions">
            <button className="btn" onClick={() => { setDraft(saved); setOpen(false) }}>Cancel</button>
            <button className="btn primary" onClick={save} disabled={!dirty}>Save</button>
          </div>
        </div>
      )}
    </div>
  )
}
