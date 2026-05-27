import { useState, useEffect, KeyboardEvent } from 'react'
import { api } from '../api/client'
import type { AppSettings } from '../types'

const PROVIDER_LABEL: Record<string, string> = {
  local: 'Local',
  anthropic: 'Anthropic',
  openai: 'OpenAI',
}

export default function SettingsView() {
  const [settings, setSettings] = useState<AppSettings | null>(null)
  const [draft, setDraft] = useState<AppSettings | null>(null)
  const [memoryContent, setMemoryContent] = useState('')
  const [memoryDraft, setMemoryDraft] = useState('')
  const [topicInput, setTopicInput] = useState('')
  const [saveState, setSaveState] = useState<'idle' | 'saving' | 'saved'>('idle')

  useEffect(() => {
    Promise.all([api.getSettings(), api.getMemory()]).then(([s, m]) => {
      setSettings(s)
      setDraft(s)
      setMemoryContent(m.content)
      setMemoryDraft(m.content)
    })
  }, [])

  const isDirty = draft && settings && (
    draft.cheapest_model_id !== settings.cheapest_model_id ||
    draft.system_prompt !== settings.system_prompt ||
    draft.memory_enabled !== settings.memory_enabled ||
    draft.guardrails_enabled !== settings.guardrails_enabled ||
    draft.guardrails_input !== settings.guardrails_input ||
    draft.guardrails_output !== settings.guardrails_output ||
    draft.guardrails_prompt !== settings.guardrails_prompt ||
    JSON.stringify(draft.guardrails_topics) !== JSON.stringify(settings.guardrails_topics) ||
    memoryDraft !== memoryContent
  )

  const handleSave = async () => {
    if (!draft) return
    setSaveState('saving')
    const [updated] = await Promise.all([
      api.updateSettings({
        cheapest_model_id: draft.cheapest_model_id,
        system_prompt: draft.system_prompt,
        memory_enabled: draft.memory_enabled,
        guardrails_enabled: draft.guardrails_enabled,
        guardrails_input: draft.guardrails_input,
        guardrails_output: draft.guardrails_output,
        guardrails_prompt: draft.guardrails_prompt,
        guardrails_topics: draft.guardrails_topics,
      }),
      api.updateMemory(memoryDraft),
    ])
    setSettings({ ...updated, models: draft.models })
    setDraft(prev => prev ? { ...prev, ...updated } : prev)
    setMemoryContent(memoryDraft)
    setSaveState('saved')
    setTimeout(() => setSaveState('idle'), 2000)
  }

  const addTopic = () => {
    const val = topicInput.trim()
    if (!val || !draft || draft.guardrails_topics.includes(val)) return
    setDraft({ ...draft, guardrails_topics: [...draft.guardrails_topics, val] })
    setTopicInput('')
  }

  const removeTopic = (topic: string) => {
    if (!draft) return
    setDraft({ ...draft, guardrails_topics: draft.guardrails_topics.filter(t => t !== topic) })
  }

  const handleTopicKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') { e.preventDefault(); addTopic() }
  }

  if (!draft) return <div className="settings-loading">Loading…</div>

  const byProvider = draft.models.reduce<Record<string, typeof draft.models>>((acc, m) => {
    ;(acc[m.provider] ??= []).push(m)
    return acc
  }, {})

  const isPromptDefault = draft.system_prompt === draft.default_system_prompt
  const isGuardPromptDefault = draft.guardrails_prompt === draft.default_guardrail_prompt

  return (
    <div className="settings-view">
      <div className="settings-header">
        <h1 className="settings-title">Settings</h1>
      </div>

      <div className="settings-body">

        {/* ── Guardrails ─────────────────────────────────────────── */}
        <section className="settings-section">
          <div className="settings-section-header">
            <h2 className="settings-section-title">Guardrails</h2>
            <p className="settings-section-desc">
              Safety checks on messages using the cheapest model. Input checks block
              before storing; output checks run after generation.
            </p>
          </div>

          <div className="settings-row">
            <div className="settings-row-label">
              <span className="settings-row-name">Enable guardrails</span>
              <span className="settings-row-hint">Master toggle for all safety checks</span>
            </div>
            <label className="toggle-switch">
              <input
                type="checkbox"
                checked={draft.guardrails_enabled}
                onChange={e => setDraft({ ...draft, guardrails_enabled: e.target.checked })}
              />
              <span className="toggle-slider" />
            </label>
          </div>

          {draft.guardrails_enabled && (<>
            <div className="settings-row">
              <div className="settings-row-label">
                <span className="settings-row-name">Check inputs</span>
                <span className="settings-row-hint">Block user messages before they're sent</span>
              </div>
              <label className="toggle-switch">
                <input
                  type="checkbox"
                  checked={draft.guardrails_input}
                  onChange={e => setDraft({ ...draft, guardrails_input: e.target.checked })}
                />
                <span className="toggle-slider" />
              </label>
            </div>

            <div className="settings-row">
              <div className="settings-row-label">
                <span className="settings-row-name">Check outputs</span>
                <span className="settings-row-hint">Block assistant responses after generation</span>
              </div>
              <label className="toggle-switch">
                <input
                  type="checkbox"
                  checked={draft.guardrails_output}
                  onChange={e => setDraft({ ...draft, guardrails_output: e.target.checked })}
                />
                <span className="toggle-slider" />
              </label>
            </div>

            {/* Denied topics */}
            <div className="settings-prompt-block">
              <label className="settings-row-name" style={{ marginBottom: 8 }}>
                Denied topics
                <span className="settings-row-hint" style={{ display: 'block', fontWeight: 400, marginTop: 2 }}>
                  Block messages about specific subjects. Leave empty for general safety only.
                </span>
              </label>

              {draft.guardrails_topics.length > 0 && (
                <div className="settings-topics-list">
                  {draft.guardrails_topics.map(t => (
                    <span key={t} className="settings-topic-tag">
                      {t}
                      <button className="settings-topic-remove" onClick={() => removeTopic(t)} title="Remove">×</button>
                    </span>
                  ))}
                </div>
              )}

              <div className="settings-topic-input-row">
                <input
                  className="settings-topic-input"
                  value={topicInput}
                  onChange={e => setTopicInput(e.target.value)}
                  onKeyDown={handleTopicKey}
                  placeholder="e.g. competitor products, internal pricing…"
                />
                <button
                  className="settings-topic-add-btn"
                  onClick={addTopic}
                  disabled={!topicInput.trim()}
                >
                  Add topic
                </button>
              </div>
            </div>

            {/* Guardrail prompt */}
            <div className="settings-prompt-block">
              <label className="settings-row-name" style={{ marginBottom: 4 }}>
                Guardrail prompt
                <span className="settings-row-hint" style={{ display: 'block', fontWeight: 400, marginTop: 2 }}>
                  System prompt for the safety classifier. Must instruct the model to return JSON.
                </span>
              </label>
              <textarea
                className="settings-prompt-textarea"
                value={draft.guardrails_prompt}
                onChange={e => setDraft({ ...draft, guardrails_prompt: e.target.value })}
                rows={7}
                spellCheck={false}
              />
              {!isGuardPromptDefault && (
                <button
                  className="settings-prompt-reset"
                  onClick={() => setDraft({ ...draft, guardrails_prompt: draft.default_guardrail_prompt })}
                >
                  Reset to default
                </button>
              )}
            </div>
          </>)}
        </section>

        {/* ── Memory ────────────────────────────────────────────── */}
        <section className="settings-section">
          <div className="settings-section-header">
            <h2 className="settings-section-title">Memory</h2>
            <p className="settings-section-desc">
              When enabled, the assistant automatically learns durable facts about you
              from your messages and uses them to personalise future responses.
            </p>
          </div>

          <div className="settings-row">
            <div className="settings-row-label">
              <span className="settings-row-name">Enable memory</span>
              <span className="settings-row-hint">Facts are extracted after each message</span>
            </div>
            <label className="toggle-switch">
              <input
                type="checkbox"
                checked={draft.memory_enabled}
                onChange={e => setDraft({ ...draft, memory_enabled: e.target.checked })}
              />
              <span className="toggle-slider" />
            </label>
          </div>

          {draft.memory_enabled && (
            <div className="settings-prompt-block">
              <label className="settings-row-name" style={{ marginBottom: 4 }}>
                Memory contents
              </label>
              <textarea
                className="settings-prompt-textarea"
                value={memoryDraft}
                onChange={e => setMemoryDraft(e.target.value)}
                rows={6}
                spellCheck={false}
                placeholder="Nothing remembered yet — the assistant will fill this in automatically."
              />
              {memoryDraft && (
                <button className="settings-prompt-reset" onClick={() => setMemoryDraft('')}>
                  Clear memory
                </button>
              )}
            </div>
          )}
        </section>

        {/* ── System prompt ──────────────────────────────────────── */}
        <section className="settings-section">
          <div className="settings-section-header">
            <h2 className="settings-section-title">System Prompt</h2>
            <p className="settings-section-desc">
              Instructions sent to the model at the start of every conversation.
              Customise the assistant's personality, tone, or domain focus.
            </p>
          </div>

          <div className="settings-prompt-block">
            <textarea
              className="settings-prompt-textarea"
              value={draft.system_prompt}
              onChange={e => setDraft({ ...draft, system_prompt: e.target.value })}
              rows={6}
              spellCheck={false}
            />
            {!isPromptDefault && (
              <button
                className="settings-prompt-reset"
                onClick={() => setDraft({ ...draft, system_prompt: draft.default_system_prompt })}
              >
                Reset to default
              </button>
            )}
          </div>
        </section>

        {/* ── Utility model ─────────────────────────────────────── */}
        <section className="settings-section">
          <div className="settings-section-header">
            <h2 className="settings-section-title">Utility Model</h2>
            <p className="settings-section-desc">
              Used for background tasks: naming conversations, updating memory, and
              running guardrail checks. Pick your cheapest model to keep costs low.
            </p>
          </div>

          <div className="settings-row">
            <div className="settings-row-label">
              <span className="settings-row-name">Cheapest model</span>
              <span className="settings-row-hint">Background tasks use this model</span>
            </div>
            <select
              className="settings-select"
              value={draft.cheapest_model_id ?? ''}
              onChange={e => setDraft({ ...draft, cheapest_model_id: e.target.value || null })}
            >
              <option value="">Auto (provider default)</option>
              {Object.entries(byProvider).map(([provider, models]) => (
                <optgroup key={provider} label={PROVIDER_LABEL[provider] ?? provider}>
                  {models.map(m => (
                    <option key={m.id} value={m.id}>
                      {m.name}{m.is_cheapest ? ' ★' : ''}
                    </option>
                  ))}
                </optgroup>
              ))}
            </select>
          </div>
        </section>

      </div>

      <div className="settings-footer">
        <button
          className="settings-save-btn"
          onClick={handleSave}
          disabled={!isDirty || saveState === 'saving'}
        >
          {saveState === 'saving' ? 'Saving…' : saveState === 'saved' ? 'Saved!' : 'Save changes'}
        </button>
      </div>
    </div>
  )
}
