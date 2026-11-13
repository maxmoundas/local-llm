import { useEffect, useRef, useState } from 'react'
import { api } from '../api/client'
import { filePreview, isPreviewableFile, usePreview } from '../lib/preview'
import { formatCost, formatDuration, liveLine } from '../lib/turn'
import type { AgentAction, InputRequest, SourceRef, TaskFile, Usage } from '../types'
import {
  CheckIcon, ChevronDownIcon, CloseIcon, CloudIcon, CodeIcon, DownloadIcon, FileIcon, GlobeIcon, HandIcon,
  EyeIcon, NewsIcon, PageIcon, PlugIcon, QuestionIcon, SlashIcon,
} from './Icons'

interface Props {
  actions: AgentAction[]
  sources: SourceRef[]
  files: TaskFile[]
  taskId?: string
  active: boolean            // the task is still working (before the answer starts)
  status?: string
  pending?: InputRequest
  startedAt?: number
  elapsedMs?: number         // saved duration, for finished turns
  usage?: Usage
  onReply?: (reply: { request_id: string; answer?: string; approved?: boolean }) => void
}

const KIND_ICON: Record<string, typeof GlobeIcon> = {
  search: GlobeIcon, news: NewsIcon, page: PageIcon, weather: CloudIcon, code: CodeIcon, file: FileIcon,
  connector: PlugIcon, question: QuestionIcon,
}

function domain(url: string): string {
  try { return new URL(url).hostname.replace(/^www\./, '') } catch { return url }
}

function plural(n: number, word: string): string {
  return `${n} ${word}${n === 1 ? '' : 's'}`
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`
}

/** Re-render every second while `on` — drives the elapsed timers. */
function useNow(on: boolean): number {
  const [now, setNow] = useState(Date.now())
  useEffect(() => {
    if (!on) return
    const id = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(id)
  }, [on])
  return now
}

const SOURCE_PREVIEW = 8

export default function Activity({
  actions, sources, files, taskId, active, status, pending, startedAt, elapsedMs, usage, onReply,
}: Props) {
  const openPreview = usePreview()
  // Open while working so each step animates in; fold away once the answer starts — unless the
  // user chose otherwise.
  const [userOpen, setUserOpen] = useState<boolean | null>(null)
  const open = userOpen ?? (active || !!pending)
  const [allSources, setAllSources] = useState(false)
  const now = useNow(active)

  const line = liveLine({ actions, status, pending })
  const elapsed = elapsedMs ?? (startedAt ? now - startedAt : undefined)
  const steps = actions.filter(a => a.kind !== 'question')
  const summary = [
    elapsed !== undefined && elapsed >= 1000 ? `Worked for ${formatDuration(elapsed)}` : null,
    plural(steps.length, 'step'),
    sources.length ? plural(sources.length, 'source') : null,
    files.length ? plural(files.length, 'file') : null,
    usage ? formatCost(usage.cost_usd) || null : null,
  ].filter(Boolean).join(' · ')

  const shown = allSources ? sources : sources.slice(0, SOURCE_PREVIEW)

  return (
    <div className={`activity${open ? ' open' : ''}${active ? ' active' : ''}`}>
      <button className="activity-header" onClick={() => setUserOpen(!open)} aria-expanded={open}>
        {active ? <span className="spinner" /> : <CheckIcon size={15} className="activity-done-icon" />}
        <span className={`activity-line${active ? ' shimmer' : ''}`} key={active ? line : 'done'}>
          {active ? line : summary}
        </span>
        {active && elapsed !== undefined && elapsed >= 2000 && (
          <span className="activity-timer">{formatDuration(elapsed)}</span>
        )}
        <ChevronDownIcon size={15} className="activity-chevron" />
      </button>

      <div className="activity-body-wrap">
        <div className="activity-body">
          {steps.length === 0 && active && <div className="activity-empty shimmer">{status ?? 'Thinking'}</div>}
          <ol className="activity-steps">
            {steps.map(a => <Step key={a.id} action={a} now={now} />)}
          </ol>

          {files.length > 0 && taskId && (
            <div className="chips">
              {files.map(f => (
                <span key={f.path} className="chip-group">
                  <a className="chip file-chip" href={api.taskFileUrl(taskId, f.path)} download>
                    <FileIcon size={14} />
                    <span className="chip-title">{f.path}</span>
                    <span className="chip-meta">{formatSize(f.size)}</span>
                    <DownloadIcon size={14} className="chip-action" />
                  </a>
                  {isPreviewableFile(f.path) && (
                    <button className="chip chip-icon" title="Preview" aria-label={`Preview ${f.path}`}
                            onClick={() => openPreview(filePreview(f.path, api.taskFileUrl(taskId, f.path)))}>
                      <EyeIcon size={14} />
                    </button>
                  )}
                </span>
              ))}
            </div>
          )}

          {sources.length > 0 && (
            <div className="chips">
              {shown.map(s => (
                <a key={s.url} className="chip" href={s.url} target="_blank" rel="noopener noreferrer" title={s.title}>
                  <span className="source-initial">{domain(s.url).charAt(0).toUpperCase()}</span>
                  <span className="chip-title">{s.title}</span>
                  <span className="chip-meta">{domain(s.url)}</span>
                </a>
              ))}
              {sources.length > SOURCE_PREVIEW && (
                <button className="chip more" onClick={() => setAllSources(a => !a)}>
                  {allSources ? 'Show fewer' : `+${sources.length - SOURCE_PREVIEW} more`}
                </button>
              )}
            </div>
          )}
        </div>
      </div>

      {pending && onReply && <InputCard request={pending} onReply={onReply} />}
    </div>
  )
}

function Step({ action, now }: { action: AgentAction; now: number }) {
  const Icon = KIND_ICON[action.kind] ?? GlobeIcon
  const running = action.status === 'running'
  const took = action.started && (action.ended ?? (running ? now : undefined))
    ? (action.ended ?? now) - action.started : undefined
  return (
    <li className={`step ${action.status}`}>
      <span className="step-icon" aria-hidden="true">
        {running && <span className="spinner small" />}
        {action.status === 'waiting' && <HandIcon size={14} />}
        {action.status === 'done' && <Icon size={14} />}
        {action.status === 'failed' && <CloseIcon size={14} />}
        {action.status === 'denied' && <SlashIcon size={14} />}
      </span>
      <span className={`step-label${running ? ' shimmer' : ''}`}>{action.label}</span>
      {action.summary && !running && <span className="step-summary">{action.summary}</span>}
      {took !== undefined && took >= 3000 && <span className="step-time">{formatDuration(took)}</span>}
    </li>
  )
}

function InputCard({ request, onReply }: {
  request: InputRequest
  onReply: (reply: { request_id: string; answer?: string; approved?: boolean }) => void
}) {
  const [answer, setAnswer] = useState('')
  const [sent, setSent] = useState(false)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  useEffect(() => { inputRef.current?.focus() }, [])

  const reply = (r: { answer?: string; approved?: boolean }) => {
    setSent(true)
    onReply({ request_id: request.id, ...r })
  }

  if (request.kind === 'approval') {
    const args = request.arguments ? JSON.stringify(request.arguments, null, 2) : ''
    return (
      <div className="input-card">
        <div className="input-card-title"><HandIcon size={15} /> Allow this action?</div>
        <div className="input-card-text">{request.label ?? request.tool}</div>
        {request.reason && <div className="input-card-reason">{request.reason}</div>}
        {args && args !== '{}' && <pre className="input-card-args">{args}</pre>}
        <div className="input-card-actions">
          <button className="btn" disabled={sent} onClick={() => reply({ approved: false })}>Deny</button>
          <button className="btn primary" disabled={sent} onClick={() => reply({ approved: true })}>Allow</button>
        </div>
      </div>
    )
  }
  return (
    <form className="input-card" onSubmit={e => { e.preventDefault(); if (answer.trim()) reply({ answer: answer.trim() }) }}>
      <div className="input-card-title"><QuestionIcon size={15} /> {request.question}</div>
      <textarea
        ref={inputRef}
        className="input-card-input"
        rows={2}
        value={answer}
        disabled={sent}
        placeholder="Your answer"
        onChange={e => setAnswer(e.target.value)}
        onKeyDown={e => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            if (answer.trim()) reply({ answer: answer.trim() })
          }
        }}
      />
      <div className="input-card-actions">
        <button type="button" className="btn" disabled={sent} onClick={() => reply({ answer: '' })}>Skip</button>
        <button type="submit" className="btn primary" disabled={sent || !answer.trim()}>Send</button>
      </div>
    </form>
  )
}
