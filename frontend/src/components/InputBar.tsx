import { useRef, useCallback } from 'react'

interface Props {
  onSend: (content: string) => void
  disabled: boolean
}

export default function InputBar({ onSend, disabled }: Props) {
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const submit = useCallback(() => {
    const el = textareaRef.current
    if (!el) return
    const value = el.value.trim()
    if (!value || disabled) return
    el.value = ''
    el.style.height = 'auto'
    onSend(value)
  }, [onSend, disabled])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  const handleInput = () => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`
  }

  return (
    <div className="input-area">
      <div className="input-row">
        <textarea
          ref={textareaRef}
          className="user-input"
          placeholder="Message Gemma…"
          rows={1}
          onKeyDown={handleKeyDown}
          onInput={handleInput}
          disabled={disabled}
        />
        <button className="send-btn" onClick={submit} disabled={disabled} title="Send (Enter)">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <line x1="22" y1="2" x2="11" y2="13"/>
            <polygon points="22 2 15 22 11 13 2 9 22 2"/>
          </svg>
        </button>
      </div>
      <div className="input-hint">Enter to send · Shift+Enter for new line</div>
    </div>
  )
}
