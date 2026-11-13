import { useEffect, useState, type ReactNode } from 'react'
import { api } from '../api/client'
import { CloseIcon, LockIcon } from './Icons'

const isMac = typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform)
export const MOD = isMac ? '⌘' : 'Ctrl'

export const SHORTCUTS: { keys: string[]; label: string }[] = [
  { keys: [MOD, 'Shift', 'O'], label: 'New chat' },
  { keys: [MOD, 'K'], label: 'Search chats' },
  { keys: [MOD, 'B'], label: 'Toggle sidebar' },
  { keys: ['Shift', 'Esc'], label: 'Focus the message box' },
  { keys: [MOD, 'Shift', 'C'], label: 'Copy the last response' },
  { keys: [MOD, ','], label: 'Settings' },
  { keys: [MOD, '/'], label: 'Show keyboard shortcuts' },
  { keys: ['Esc'], label: 'Close preview or dialog' },
  { keys: ['Enter'], label: 'Send · Shift+Enter for a new line' },
]

export function Modal({ title, onClose, children }: { title: string; onClose: () => void; children: ReactNode }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])
  return (
    <div className="modal-scrim" onMouseDown={e => { if (e.target === e.currentTarget) onClose() }}>
      <div className="modal" role="dialog" aria-modal="true" aria-label={title}>
        <header className="modal-head">
          <h2>{title}</h2>
          <button className="icon-btn small" onClick={onClose} aria-label="Close"><CloseIcon size={15} /></button>
        </header>
        {children}
      </div>
    </div>
  )
}

export function ShortcutsDialog({ onClose }: { onClose: () => void }) {
  return (
    <Modal title="Keyboard shortcuts" onClose={onClose}>
      <ul className="shortcut-list">
        {SHORTCUTS.map(s => (
          <li key={s.label}>
            <span>{s.label}</span>
            <span className="keys">{s.keys.map(k => <kbd key={k}>{k}</kbd>)}</span>
          </li>
        ))}
      </ul>
    </Modal>
  )
}

export function NewProjectDialog({ onClose, onCreate }: { onClose: () => void; onCreate: (name: string, description: string) => Promise<void> }) {
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [busy, setBusy] = useState(false)
  const submit = async () => {
    if (!name.trim()) return
    setBusy(true)
    await onCreate(name.trim(), description.trim())
    setBusy(false)
  }
  return (
    <Modal title="New project" onClose={onClose}>
      <p className="setting-hint">A project groups chats that share instructions, files, and their own memory.</p>
      <label className="field">
        <span className="field-label">Name</span>
        <input className="text-input wide" autoFocus value={name} onChange={e => setName(e.target.value)}
               onKeyDown={e => { if (e.key === 'Enter') submit() }} placeholder="e.g. Kitchen renovation" />
      </label>
      <label className="field">
        <span className="field-label">What is it about? <span className="muted-text">(optional)</span></span>
        <input className="text-input wide" value={description} onChange={e => setDescription(e.target.value)}
               onKeyDown={e => { if (e.key === 'Enter') submit() }} />
      </label>
      <div className="form-actions">
        <button className="btn" onClick={onClose}>Cancel</button>
        <button className="btn primary" onClick={submit} disabled={!name.trim() || busy}>Create project</button>
      </div>
    </Modal>
  )
}

export function LoginScreen({ onDone }: { onDone: () => void }) {
  const [token, setToken] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const submit = async () => {
    setBusy(true)
    setError(null)
    try {
      await api.login(token)
      onDone()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Sign-in failed')
    }
    setBusy(false)
  }
  return (
    <div className="login-screen">
      <div className="login-card">
        <LockIcon size={26} className="login-icon" />
        <h1>Local LLM</h1>
        <p className="setting-hint">This server needs its access token (<code>ACCESS_TOKEN</code> in <code>backend/.env</code>).</p>
        <input className="text-input wide" type="password" autoFocus value={token} placeholder="Access token"
               onChange={e => setToken(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && token) submit() }} />
        {error && <p className="form-error">{error}</p>}
        <button className="btn primary" onClick={submit} disabled={!token || busy}>{busy ? 'Signing in…' : 'Sign in'}</button>
      </div>
    </div>
  )
}
