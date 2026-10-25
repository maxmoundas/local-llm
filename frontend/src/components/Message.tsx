import { memo, useState } from 'react'
import type { ResearchStep, SourceRef } from '../types'
import { CheckIcon, CopyIcon } from './Icons'
import Markdown from './Markdown'
import ResearchSteps from './ResearchSteps'

export interface DisplayMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  imageDataUrl?: string
  steps?: ResearchStep[]
  sources?: SourceRef[]
  status?: string        // live research status (only while streaming)
  streaming?: boolean
  stopped?: boolean
  notice?: { kind: 'error' | 'blocked'; text: string }
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

function Message({ msg, isLast }: { msg: DisplayMessage; isLast: boolean }) {
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

  const researching = !!msg.streaming && !msg.content
  // Show the block while a live status is running, and afterwards only if a tool actually ran.
  const hasResearch = (msg.steps?.length ?? 0) > 0 || (researching && !!msg.status)
  const waiting = researching && !hasResearch

  return (
    <div className="msg assistant">
      {hasResearch && (
        <ResearchSteps steps={msg.steps ?? []} sources={msg.sources ?? []} status={msg.status} active={researching} />
      )}
      {msg.content && (
        <div className="prose">
          <Markdown content={msg.content} />
        </div>
      )}
      {waiting && <div className="thinking-dot" aria-label="Thinking" />}
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
