import { memo, useState } from 'react'
import type { AgentAction, InputRequest, SourceRef, TaskFile, Usage } from '../types'
import Activity from './Activity'
import { CheckIcon, CopyIcon } from './Icons'
import Markdown from './Markdown'

export interface DisplayMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  imageDataUrl?: string
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
}

type ReplyFn = (reply: { request_id: string; answer?: string; approved?: boolean }) => void

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

function Message({ msg, isLast, onReply }: { msg: DisplayMessage; isLast: boolean; onReply?: ReplyFn }) {
  if (msg.role === 'user') {
    return (
      <div className="msg user">
        <div className="user-bubble">
          {msg.imageDataUrl && <img className="user-image" src={msg.imageDataUrl} alt="attached" />}
          {msg.content && <div className="user-text">{msg.content}</div>}
        </div>
        {msg.content && <div className="msg-actions"><CopyButton text={msg.content} /></div>}
      </div>
    )
  }

  const working = !!msg.streaming && !msg.content
  // The activity card appears once a tool runs (or the agent asks something). Before that, and for
  // plain answers, a single quiet status line is enough.
  const hasActivity = (msg.actions?.length ?? 0) > 0 || !!msg.pending
  const showLine = working && !hasActivity

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
          onReply={onReply}
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
      {msg.notice && <div className={`notice ${msg.notice.kind}`}>{msg.notice.text}</div>}
      {msg.stopped && !msg.streaming && <div className="stopped-note">Stopped</div>}
      {!msg.streaming && msg.content && (
        <div className={`msg-actions${isLast ? ' visible' : ''}`}>
          <CopyButton text={msg.content} />
        </div>
      )}
    </div>
  )
}

export default memo(Message)
