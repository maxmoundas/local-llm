import { useState, useRef, useEffect, memo } from 'react'
import ReactMarkdown from 'react-markdown'
import rehypeHighlight from 'rehype-highlight'
import 'highlight.js/styles/github-dark.css'

interface Props {
  role: 'user' | 'assistant'
  content: string
  streaming?: boolean
  imageDataUrl?: string
}

function CopyIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
    </svg>
  )
}

function CheckIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12"/>
    </svg>
  )
}

function Message({ role, content, streaming, imageDataUrl }: Props) {
  const endRef = useRef<HTMLDivElement>(null)
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'instant' })
  }, [content])

  function handleCopy() {
    navigator.clipboard.writeText(content)
    setCopied(true)
    setTimeout(() => setCopied(false), 1800)
  }

  return (
    <div className={`message ${role}`}>
      {role === 'assistant' && <div className="avatar">G</div>}

      <div className="bubble-outer">
        <div className="bubble">
          {imageDataUrl && role === 'user' && (
            <img className="bubble-image" src={imageDataUrl} alt="attached" />
          )}

          {role === 'assistant' ? (
            <>
              <ReactMarkdown rehypePlugins={[rehypeHighlight]}>{content}</ReactMarkdown>
              {streaming && <span className="typing-cursor" />}
            </>
          ) : (
            content && <span>{content}</span>
          )}

          <div ref={endRef} />
        </div>

        {!streaming && content && (
          <div className="bubble-actions">
            <button
              className={`action-btn${copied ? ' copied' : ''}`}
              onClick={handleCopy}
              title="Copy message"
            >
              {copied ? <CheckIcon /> : <CopyIcon />}
              <span>{copied ? 'Copied' : 'Copy'}</span>
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

export default memo(Message)
