import { useEffect, useState, type KeyboardEvent } from 'react'
import { Row, Segmented, Toggle } from './controls'
import UsageTab from './UsageTab'
import { api } from '../api/client'
import { getThemePref, setThemePref, type ThemePref } from '../lib/theme'
import type { AppSettings } from '../types'

type Editable = Omit<AppSettings, 'models' | 'jev_configured' | 'default_system_prompt' | 'default_guardrail_prompt'
  | 'sandbox_active' | 'connectors' | 'transcription_available' | 'access_token_set'>
type Tab = 'general' | 'usage' | 'research' | 'jev' | 'safety' | 'memory' | 'models'

const TABS: { id: Tab; label: string }[] = [
  { id: 'general', label: 'General' },
  { id: 'usage', label: 'Usage' },
  { id: 'research', label: 'Agent' },
  { id: 'jev', label: 'Jev' },
  { id: 'safety', label: 'Safety' },
  { id: 'memory', label: 'Memory' },
  { id: 'models', label: 'Models' },
]

const PROVIDER_LABEL: Record<string, string> = {
  local: 'Local', anthropic: 'Anthropic', openai: 'OpenAI', ollama: 'Ollama', lmstudio: 'LM Studio',
}

const SANDBOX_LABEL: Record<string, string> = { docker: 'Docker sandbox', local: 'Local process (not isolated)', off: 'Off' }

function editable(s: AppSettings): Editable {
  const {
    models: _m, jev_configured: _j, default_system_prompt: _d, default_guardrail_prompt: _g,
    sandbox_active: _sa, connectors: _c, transcription_available: _t, access_token_set: _a, ...rest
  } = s
  return rest
}

export default function SettingsView({ onDisplayName }: { onDisplayName: (name: string) => void }) {
  const [settings, setSettings] = useState<AppSettings | null>(null)
  const [draft, setDraft] = useState<Editable | null>(null)
  const [memory, setMemory] = useState('')
  const [memoryDraft, setMemoryDraft] = useState('')
  const [topicInput, setTopicInput] = useState('')
  const [tab, setTab] = useState<Tab>('general')
  const [theme, setTheme] = useState<ThemePref>(getThemePref)
  const [saveState, setSaveState] = useState<'idle' | 'saving' | 'saved'>('idle')

  useEffect(() => {
    Promise.all([api.getSettings(), api.getMemory()]).then(([s, m]) => {
      setSettings(s)
      setDraft(editable(s))
      setMemory(m.content)
      setMemoryDraft(m.content)
    })
  }, [])

  if (!settings || !draft) return <div className="settings-loading">Loading…</div>

  const set = <K extends keyof Editable>(key: K, value: Editable[K]) => setDraft({ ...draft, [key]: value })
  const dirty = JSON.stringify(draft) !== JSON.stringify(editable(settings)) || memoryDraft !== memory

  const save = async () => {
    setSaveState('saving')
    // Only write memory if it was edited here — it may have been updated in the background since load.
    const [updated] = await Promise.all([
      api.updateSettings(draft),
      memoryDraft !== memory ? api.updateMemory(memoryDraft) : null,
    ])
    const merged = { ...settings, ...updated }
    setSettings(merged)
    setDraft(editable(merged))
    setMemory(memoryDraft)
    onDisplayName(updated.display_name ?? '')
    setSaveState('saved')
    setTimeout(() => setSaveState('idle'), 1800)
  }

  const addTopic = () => {
    const t = topicInput.trim()
    if (t && !draft.guardrails_topics.includes(t)) set('guardrails_topics', [...draft.guardrails_topics, t])
    setTopicInput('')
  }

  const jevStatus = !settings.jev_configured
    ? { cls: 'off', text: 'No API key' }
    : draft.jev_enabled ? { cls: 'on', text: 'Connected' } : { cls: 'off', text: 'Turned off' }
  const jevActive = settings.jev_configured && draft.jev_enabled

  const byProvider = settings.models.reduce<Record<string, typeof settings.models>>((acc, m) => {
    ;(acc[m.provider] ??= []).push(m)
    return acc
  }, {})

  return (
    <div className="settings">
      <div className="settings-scroll">
        <div className="settings-inner">
          <h1 className="settings-title">Settings</h1>
          <div className="settings-layout">
            <nav className="settings-tabs">
              {TABS.map(t => (
                <button key={t.id} className={tab === t.id ? 'active' : ''} onClick={() => setTab(t.id)}>{t.label}</button>
              ))}
            </nav>

            <div className="settings-panel">
              {tab === 'general' && (<>
                <Row name="What should we call you?" hint="Used in the greeting on new chats.">
                  <input className="text-input" value={draft.display_name}
                         onChange={e => set('display_name', e.target.value)} placeholder="Your name" />
                </Row>
                <Row name="Appearance" hint="System follows your OS light/dark setting.">
                  <Segmented value={theme} onChange={v => { setTheme(v); setThemePref(v) }} options={[
                    { value: 'system', label: 'System' }, { value: 'light', label: 'Light' }, { value: 'dark', label: 'Dark' },
                  ]} />
                </Row>
                <Row stacked name="System prompt" hint="Instructions sent at the start of every conversation.">
                  <textarea className="text-area" rows={7} value={draft.system_prompt}
                            onChange={e => set('system_prompt', e.target.value)} />
                  {draft.system_prompt !== settings.default_system_prompt && (
                    <button className="link-btn" onClick={() => set('system_prompt', settings.default_system_prompt)}>Reset to default</button>
                  )}
                </Row>
                <Row name="Voice input" hint={settings.transcription_available
                  ? 'Speech is transcribed on this machine with Whisper. Larger models are more accurate and slower; the first use downloads the model.'
                  : <>Install <code>faster-whisper</code> in the backend (<code>pip install faster-whisper</code>) and restart to show the microphone button.</>}>
                  <select className="select" value={draft.whisper_model} disabled={!settings.transcription_available}
                          onChange={e => set('whisper_model', e.target.value as Editable['whisper_model'])}>
                    {(['tiny', 'base', 'small', 'medium', 'large-v3'] as const).map(m => <option key={m} value={m}>{m}</option>)}
                  </select>
                </Row>
                <Row name="Network access" hint={settings.access_token_set
                  ? 'An access token is set: other devices can use the app after signing in with it (if the backend listens on your network).'
                  : <>Only this computer can use the app. To allow other devices, set <code>ACCESS_TOKEN</code> in <code>backend/.env</code> and <code>server.host: 0.0.0.0</code> in <code>config.yaml</code>.</>}>
                  <span className={`status-pill ${settings.access_token_set ? 'on' : 'off'}`}>{settings.access_token_set ? 'Token required' : 'This computer only'}</span>
                </Row>
              </>)}

              {tab === 'usage' && (
                <UsageTab budget={draft.monthly_budget_usd} onBudget={v => set('monthly_budget_usd', v)} />
              )}

              {tab === 'research' && (<>
                <p className="settings-intro">
                  When a message needs more than a direct answer, the assistant works on it step by step in the
                  background: searching and reading the web, running code, working with files, and using connected
                  services. You see a one-line update for each step. Work continues if you close the tab.
                </p>
                <Row name="Agent" hint="Turn off to always answer directly from the model.">
                  <Toggle checked={draft.agent_enabled} onChange={v => set('agent_enabled', v)} />
                </Row>
                <Row name="Maximum steps" hint="Each step can run several tools in parallel.">
                  <select className="select" value={draft.agent_max_steps} disabled={!draft.agent_enabled}
                          onChange={e => set('agent_max_steps', Number(e.target.value))}>
                    {[2, 4, 6, 8, 12, 16, 24, 32, 40].map(n => <option key={n} value={n}>{n}</option>)}
                  </select>
                </Row>
                <Row name="Spending limit per task" hint="Stops the tool loop once a task has spent this much on API calls, then answers with what it has.">
                  <select className="select" value={String(draft.task_budget_usd ?? 0)} disabled={!draft.agent_enabled}
                          onChange={e => set('task_budget_usd', Number(e.target.value) || null)}>
                    {[0.05, 0.1, 0.25, 0.5, 1, 2, 5].map(n => <option key={n} value={String(n)}>${n.toFixed(2)}</option>)}
                    <option value="0">No limit</option>
                  </select>
                </Row>
                <Row name="Time limit per task">
                  <select className="select" value={String(draft.task_max_minutes ?? 0)} disabled={!draft.agent_enabled}
                          onChange={e => set('task_max_minutes', Number(e.target.value) || null)}>
                    {[2, 5, 10, 20, 30, 60, 120].map(n => <option key={n} value={String(n)}>{n} min</option>)}
                    <option value="0">No limit</option>
                  </select>
                </Row>
                <Row name="Model for agent steps" hint="Picks tools between steps. A cheap model here keeps long tasks cheap; the chat model still writes the answer.">
                  <select className="select" value={draft.agent_model_id ?? ''} disabled={!draft.agent_enabled}
                          onChange={e => set('agent_model_id', e.target.value || null)}>
                    <option value="">Same as chat model</option>
                    {Object.entries(byProvider).map(([provider, ms]) => (
                      <optgroup key={provider} label={PROVIDER_LABEL[provider] ?? provider}>
                        {ms.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}
                      </optgroup>
                    ))}
                  </select>
                </Row>

                <h2 className="settings-subhead">Code</h2>
                <Row name="Run code" hint={`Now: ${SANDBOX_LABEL[settings.sandbox_active] ?? settings.sandbox_active}. Auto uses Docker when it's running. "Local" runs code directly on this machine — only use it if you trust every page the agent reads.`}>
                  <Segmented value={draft.code_execution} onChange={v => set('code_execution', v)}
                             options={[{ value: 'auto', label: 'Auto' }, { value: 'docker', label: 'Docker' },
                                       { value: 'local', label: 'Local' }, { value: 'off', label: 'Off' }]} />
                </Row>
                <Row name="Internet access for code" hint="Lets sandboxed code install packages and call APIs.">
                  <Toggle checked={draft.sandbox_network} disabled={draft.code_execution === 'off'}
                          onChange={v => set('sandbox_network', v)} />
                </Row>

                <h2 className="settings-subhead">Connectors</h2>
                {Object.keys(settings.connectors ?? {}).length === 0 ? (
                  <p className="setting-hint">
                    None yet. Add MCP servers (email, calendar, GitHub, files…) to <code>backend/connectors.yaml</code> —
                    see <code>connectors.example.yaml</code> — and restart the backend.
                  </p>
                ) : Object.entries(settings.connectors).map(([name, state]) => (
                  <Row key={name} name={name}><span className="muted-text">{state}</span></Row>
                ))}

                <Row name="Router" hint={jevActive && draft.jev_routing
                  ? 'Jev decides when a message needs tools, and when research findings are enough.'
                  : 'The utility model decides when a message needs tools; the agent decides when to stop.'}>
                  <span className="muted-text">{jevActive && draft.jev_routing ? 'Jev' : 'Utility model'}</span>
                </Row>
              </>)}

              {tab === 'jev' && (<>
                <p className="settings-intro">
                  Jev is TypeSafe&apos;s System One model: instead of writing text, it answers typed yes/no, choice and
                  score questions in well under a second. It handles the app&apos;s quick decisions. Anything you turn
                  off here falls back to the utility model.
                </p>
                <Row name="Status" hint={settings.jev_configured ? undefined : <>Add <code>TYPESAFE_API_KEY</code> to <code>backend/.env</code> and restart.</>}>
                  <span className={`status-pill ${jevStatus.cls}`}>{jevStatus.text}</span>
                </Row>
                <Row name="Use Jev" hint="Master switch for every Jev decision.">
                  <Toggle checked={draft.jev_enabled} disabled={!settings.jev_configured} onChange={v => set('jev_enabled', v)} />
                </Row>
                <Row name="Model">
                  <select className="select" value={draft.jev_model} disabled={!jevActive}
                          onChange={e => set('jev_model', e.target.value)}>
                    <option value="jev-latest">jev-latest</option>
                    <option value="jev-preview">jev-preview</option>
                  </select>
                </Row>
                <h2 className="settings-subhead">Use Jev for</h2>
                <Row name="Agent routing" hint="Whether a message needs tools, and whether research findings are enough.">
                  <Toggle checked={draft.jev_routing} disabled={!jevActive} onChange={v => set('jev_routing', v)} />
                </Row>
                <Row name="Safety checks" hint="Hazard probabilities and harm severity for each message.">
                  <Toggle checked={draft.jev_guardrails} disabled={!jevActive} onChange={v => set('jev_guardrails', v)} />
                </Row>
                <Row name="Memory gating" hint="Skips memory extraction on messages with nothing worth remembering.">
                  <Toggle checked={draft.jev_memory} disabled={!jevActive} onChange={v => set('jev_memory', v)} />
                </Row>
              </>)}

              {tab === 'safety' && (<>
                <Row name="Safety checks" hint="Screen messages for jailbreaks, harmful requests, and denied topics.">
                  <Toggle checked={draft.guardrails_enabled} onChange={v => set('guardrails_enabled', v)} />
                </Row>
                {draft.guardrails_enabled && (<>
                  <Row name="Check my messages" hint="Blocked messages are never sent or saved.">
                    <Toggle checked={draft.guardrails_input} onChange={v => set('guardrails_input', v)} />
                  </Row>
                  <Row name="Check responses" hint="Runs after each reply; adds a little latency.">
                    <Toggle checked={draft.guardrails_output} onChange={v => set('guardrails_output', v)} />
                  </Row>
                  {jevActive && draft.jev_guardrails && (
                    <Row name="Policy" hint="Permissive only blocks when Jev is very sure.">
                      <Segmented value={draft.guardrails_policy} onChange={v => set('guardrails_policy', v)} options={[
                        { value: 'strict', label: 'Strict' }, { value: 'permissive', label: 'Permissive' },
                      ]} />
                    </Row>
                  )}
                  <Row stacked name="Denied topics" hint="Also block anything about these subjects.">
                    {draft.guardrails_topics.length > 0 && (
                      <div className="tags">
                        {draft.guardrails_topics.map(t => (
                          <span key={t} className="tag">
                            {t}
                            <button onClick={() => set('guardrails_topics', draft.guardrails_topics.filter(x => x !== t))}
                                    aria-label={`Remove ${t}`}>×</button>
                          </span>
                        ))}
                      </div>
                    )}
                    <div className="inline-input">
                      <input className="text-input" value={topicInput} placeholder="e.g. competitor products"
                             onChange={e => setTopicInput(e.target.value)}
                             onKeyDown={(e: KeyboardEvent<HTMLInputElement>) => { if (e.key === 'Enter') { e.preventDefault(); addTopic() } }} />
                      <button className="btn" onClick={addTopic} disabled={!topicInput.trim()}>Add</button>
                    </div>
                  </Row>
                  <details className="advanced">
                    <summary>Fallback classifier prompt</summary>
                    <p className="setting-hint">Used when Jev is off or unavailable. Must tell the model to return JSON.</p>
                    <textarea className="text-area mono" rows={9} value={draft.guardrails_prompt}
                              onChange={e => set('guardrails_prompt', e.target.value)} />
                    {draft.guardrails_prompt !== settings.default_guardrail_prompt && (
                      <button className="link-btn" onClick={() => set('guardrails_prompt', settings.default_guardrail_prompt)}>Reset to default</button>
                    )}
                  </details>
                </>)}
              </>)}

              {tab === 'memory' && (<>
                <Row name="Memory" hint="Learn lasting facts about you from your messages and use them in future chats.">
                  <Toggle checked={draft.memory_enabled} onChange={v => set('memory_enabled', v)} />
                </Row>
                <Row stacked name="What I remember"
                     hint={draft.memory_enabled
                       ? 'One fact per line. Edit freely — this is sent with every conversation.'
                       : "Memory is off: these facts are kept but not sent to chats or updated. Clear them if you don't want them stored."}>
                  <textarea className="text-area" rows={8} value={memoryDraft} onChange={e => setMemoryDraft(e.target.value)}
                            placeholder="Nothing remembered yet." />
                  {memoryDraft && <button className="link-btn" onClick={() => setMemoryDraft('')}>Clear memory</button>}
                </Row>
              </>)}

              {tab === 'models' && (
                <Row name="Utility model" hint="Writes chat titles, updates memory, and makes any decision Jev isn't handling. The agent uses the chat model unless you pick one under Agent.">
                  <select className="select" value={draft.cheapest_model_id ?? ''}
                          onChange={e => set('cheapest_model_id', e.target.value || null)}>
                    <option value="">Auto (cheapest available)</option>
                    {Object.entries(byProvider).map(([provider, ms]) => (
                      <optgroup key={provider} label={PROVIDER_LABEL[provider] ?? provider}>
                        {ms.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}
                      </optgroup>
                    ))}
                  </select>
                </Row>
              )}
            </div>
          </div>
        </div>
      </div>

      {(dirty || saveState !== 'idle') && (
        <div className="settings-savebar">
          <span>{saveState === 'saved' ? 'Saved' : 'You have unsaved changes'}</span>
          {saveState !== 'saved' && (
            <>
              <button className="btn" onClick={() => { setDraft(editable(settings)); setMemoryDraft(memory) }}>Discard</button>
              <button className="btn primary" onClick={save} disabled={saveState === 'saving'}>
                {saveState === 'saving' ? 'Saving…' : 'Save changes'}
              </button>
            </>
          )}
        </div>
      )}
    </div>
  )
}
