import { memo, useState } from 'react'
import { api, formatSize } from '../api/client'
import { filePreview, isPreviewableFile, usePreview } from '../lib/preview'
import type { AgentAction, Attachment, InputRequest, MemoryChange, SourceRef, TaskFile, Usage } from '../types'
import Activity from './Activity'
import { CheckIcon, ChevronLeftIcon, ChevronRightIcon, CopyIcon, DownloadIcon, EyeIcon, FileIcon, PencilIcon, RefreshIcon } from './Icons'
import Markdown from './Markdown'

export interface DisplayMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  index?: number            // position in the saved chat (absent for unsaved turns, e.g. errors)
  attachments?: Attachment[]
  localImages?: string[]    // data URLs shown before the upload is saved
  versionCount?: number
  versionIndex?: number
  modelId?: string
  actions?: AgentAction[]
  sources?: SourceRef[]
  files?: TaskFile[]
  taskId?: string
  status?: string          // live one-line status (only while streaming)
  pending?: InputRequest   // the agent is waiting on the user
  startedAt?: number       // live turns: when the turn began (drives the timer)
  elapsedMs?: number       // saved turns: how long the work took
  usage?: Usage
  streaming?: boolean
  stopped?: boolean
  notice?: { kind: 'error' | 'blocked'; text: string }
  memory?: MemoryChange
  retryable?: boolean      // an unsaved failed turn whose question was saved: Retry answers it again
}

type ReplyFn = (reply: { request_id: string; answer?: string; approved?: boolean }) => void

export interface MessageHandlers {
  onReply?: ReplyFn
  onEdit?: (index: number, content: string) => void
  onRegenerate?: (index: number | null) => void
  onSwitchVersion?: (index: number, version: number) => void
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <button
      className="icon-btn small"
      onClick={() => {
        navigator.clipboard.writeText(text)
        setCopied(true)
        setTimeout(() => setCopied(false), 1600)
      }}
      title={copied ? 'Copied' : 'Copy'}
      aria-label="Copy message"
    >
      {copied ? <CheckIcon size={15} /> : <CopyIcon size={15} />}
    </button>
  )
}

/** ‹ 2 / 3 › — switch between versions of a message (from regenerate or edit). */
function VersionSwitch({ msg, onSwitch, disabled }: { msg: DisplayMessage; onSwitch?: MessageHandlers['onSwitchVersion']; disabled: boolean }) {
  if (!msg.versionCount || msg.versionCount < 2 || msg.index === undefined || !onSwitch) return null
  const cur = msg.versionIndex ?? msg.versionCount - 1
  const go = (v: number) => onSwitch(msg.index!, v)
  return (
    <span className="version-switch">
      <button className="icon-btn small" disabled={disabled || cur === 0} onClick={() => go(cur - 1)} aria-label="Previous version">
        <ChevronLeftIcon size={14} />
      </button>
      <span className="version-label">{cur + 1} / {msg.versionCount}</span>
      <button className="icon-btn small" disabled={disabled || cur === msg.versionCount - 1} onClick={() => go(cur + 1)} aria-label="Next version">
        <ChevronRightIcon size={14} />
      </button>
    </span>
  )
}

/** "Remembered: X" / "Forgot: Y" under a reply whose message changed the user's memory. */
function MemoryNote({ change }: { change: MemoryChange }) {
  const parts = [
    change.added.length ? `Remembered: ${change.added.join('; ')}` : '',
    change.removed.length ? `Forgot: ${change.removed.join('; ')}` : '',
  ].filter(Boolean)
  if (!parts.length) return null
  return (
    <div className="memory-note" title={`${parts.join('\n')}\n\nEdit in Settings → Memory`}>
      <span className="memory-note-label">Memory updated</span>
      <span className="memory-note-text">{parts.join(' · ')}</span>
    </div>
  )
}

function UserAttachments({ chatId, msg }: { chatId: string; msg: DisplayMessage }) {
  const openPreview = usePreview()
  const images = (msg.attachments ?? []).filter(a => a.kind === 'image')
  const docs = (msg.attachments ?? []).filter(a => a.kind === 'document')
  if (!images.length && !docs.length && !msg.localImages?.length) return null
  return (
    <div className="user-attachments">
      {msg.localImages?.map((src, i) => <img key={`l${i}`} className="user-image" src={src} alt="attached" />)}
      {images.map(a => (
        <button key={a.id} className="user-image-btn" onClick={() => openPreview({ kind: 'image', title: a.name, url: api.fileUrl(chatId, a.id) })}>
          <img className="user-image" src={api.fileUrl(chatId, a.id)} alt={a.name} loading="lazy" />
        </button>
      ))}
      {docs.map(a => (
        <a key={a.id} className="doc-chip" href={api.fileUrl(chatId, a.id, true)} download={a.name} title={`Download ${a.name}`}>
          <span className="doc-chip-icon"><FileIcon size={16} /></span>
          <span className="doc-chip-text">
            <span className="doc-chip-name">{a.name}</span>
            <span className="doc-chip-meta">{a.name.split('.').pop()?.toUpperCase()} · {formatSize(a.size)}</span>
          </span>
        </a>
      ))}
    </div>
  )
}

/** Files the agent made: images inline, the rest as chips with Preview and Download. */
function OutputFiles({ taskId, files }: { taskId: string; files: TaskFile[] }) {
  const openPreview = usePreview()
  const images = files.filter(f => /\.(png|jpe?g|gif|webp|svg)$/i.test(f.path))
  const others = files.filter(f => !images.includes(f))
  return (
    <div className="output-files">
      {images.length > 0 && (
        <div className="output-images">
          {images.map(f => {
            const url = api.taskFileUrl(taskId, f.path)
            return (
              <button key={f.path} className="output-image" onClick={() => openPreview({ kind: 'image', title: f.path, url })}>
                <img src={url} alt={f.path} loading="lazy" />
              </button>
            )
          })}
        </div>
      )}
      {others.length > 0 && (
        <div className="chips">
          {others.map(f => {
            const url = api.taskFileUrl(taskId, f.path)
            return (
              <span key={f.path} className="chip-group">
                {isPreviewableFile(f.path) ? (
                  <button className="chip file-chip" onClick={() => openPreview(filePreview(f.path, url))}>
                    <EyeIcon size={14} /><span className="chip-title">{f.path}</span>
                    <span className="chip-meta">{formatSize(f.size)}</span>
                  </button>
                ) : (
                  <span className="chip file-chip"><FileIcon size={14} /><span className="chip-title">{f.path}</span>
                    <span className="chip-meta">{formatSize(f.size)}</span></span>
                )}
                <a className="chip chip-icon" href={url} download title="Download" aria-label={`Download ${f.path}`}>
                  <DownloadIcon size={14} />
                </a>
              </span>
            )
          })}
        </div>
      )}
    </div>
  )
}

function UserMessage({ chatId, msg, busy, handlers }: { chatId: string; msg: DisplayMessage; busy: boolean; handlers: MessageHandlers }) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(msg.content)
  const canEdit = !!handlers.onEdit && msg.index !== undefined && !busy

  if (editing) {
    const save = () => {
      if (!draft.trim() && !msg.attachments?.length) return
      setEditing(false)
      handlers.onEdit!(msg.index!, draft.trim())
    }
    return (
      <div className="msg user">
        <div className="user-edit">
          <textarea className="text-area" autoFocus value={draft} rows={Math.min(12, draft.split('\n').length + 1)}
                    onChange={e => setDraft(e.target.value)}
                    onKeyDown={e => {
                      if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); save() }
                      if (e.key === 'Escape') { setDraft(msg.content); setEditing(false) }
                    }} />
          <div className="user-edit-actions">
            <span className="muted-text">Sending creates a new version; the original stays one click away.</span>
            <button className="btn" onClick={() => { setDraft(msg.content); setEditing(false) }}>Cancel</button>
            <button className="btn primary" onClick={save} disabled={!draft.trim() && !msg.attachments?.length}>Send</button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="msg user">
      <UserAttachments chatId={chatId} msg={msg} />
      {msg.content && <div className="user-bubble"><div className="user-text">{msg.content}</div></div>}
      <div className="msg-actions">
        <VersionSwitch msg={msg} onSwitch={handlers.onSwitchVersion} disabled={busy} />
        {canEdit && (
          <button className="icon-btn small" onClick={() => { setDraft(msg.content); setEditing(true) }} title="Edit" aria-label="Edit message">
            <PencilIcon size={15} />
          </button>
        )}
        {msg.content && <CopyButton text={msg.content} />}
      </div>
    </div>
  )
}

function Message({ chatId, msg, isLast, busy, handlers }: {
  chatId: string; msg: DisplayMessage; isLast: boolean; busy: boolean; handlers: MessageHandlers
}) {
  if (msg.role === 'user') return <UserMessage chatId={chatId} msg={msg} busy={busy} handlers={handlers} />

  const working = !!msg.streaming && !msg.content
  // The activity card appears once a tool runs (or the agent asks something). Before that, and for
  // plain answers, a single quiet status line is enough.
  const hasActivity = (msg.actions?.length ?? 0) > 0 || !!msg.pending
  const showLine = working && !hasActivity
  // Saved replies can be regenerated; an unsaved failed turn can be retried (but not a blocked one).
  const canRegenerate = !!handlers.onRegenerate && !msg.streaming && !busy
    && (msg.index !== undefined || (isLast && !!msg.retryable))

  return (
    <div className="msg assistant">
      {hasActivity && (
        <Activity
          actions={msg.actions ?? []}
          sources={msg.sources ?? []}
          files={msg.files ?? []}
          taskId={msg.taskId}
          active={working}
          status={msg.status}
          pending={msg.pending}
          startedAt={msg.startedAt}
          elapsedMs={msg.streaming ? undefined : msg.elapsedMs}
          usage={msg.streaming ? undefined : msg.usage}
          onReply={handlers.onReply}
        />
      )}
      {showLine && (
        <div className="status-line" aria-live="polite">
          <span className="spinner small" />
          <span className="shimmer" key={msg.status ?? 'thinking'}>{msg.status ?? 'Thinking'}</span>
        </div>
      )}
      {msg.content && (
        <div className="prose">
          <Markdown content={msg.content} />
        </div>
      )}
      {msg.streaming && msg.content && <span className="stream-dot" aria-hidden="true" />}
      {!msg.streaming && msg.taskId && (msg.files?.length ?? 0) > 0 && <OutputFiles taskId={msg.taskId} files={msg.files!} />}
      {msg.notice && <div className={`notice ${msg.notice.kind}`}>{msg.notice.text}</div>}
      {msg.stopped && !msg.streaming && <div className="stopped-note">Stopped</div>}
      {msg.memory && <MemoryNote change={msg.memory} />}
      {!msg.streaming && (msg.content || msg.notice || (msg.versionCount ?? 0) > 1) && (
        <div className={`msg-actions${isLast ? ' visible' : ''}`}>
          <VersionSwitch msg={msg} onSwitch={handlers.onSwitchVersion} disabled={busy} />
          {msg.content && <CopyButton text={msg.content} />}
          {canRegenerate && (
            <button className="icon-btn small" onClick={() => handlers.onRegenerate!(msg.index ?? null)}
                    title="Retry" aria-label="Regenerate response">
              <RefreshIcon size={15} />
            </button>
          )}
        </div>
      )}
    </div>
  )
}

export default memo(Message)
