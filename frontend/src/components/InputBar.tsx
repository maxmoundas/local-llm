import { useRef, useCallback, useState } from 'react'
import type { AttachedImage } from '../types'

interface Props {
  onSend: (content: string, image?: AttachedImage) => void
  disabled: boolean
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

async function fileToAttached(file: File): Promise<AttachedImage> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      const dataUrl = reader.result as string
      resolve({
        dataUrl,
        base64: dataUrl.split(',')[1],
        name: file.name,
        size: file.size,
      })
    }
    reader.onerror = reject
    reader.readAsDataURL(file)
  })
}

export default function InputBar({ onSend, disabled }: Props) {
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [attached, setAttached] = useState<AttachedImage | null>(null)

  const handleFile = useCallback(async (file: File) => {
    if (!file.type.startsWith('image/')) return
    const img = await fileToAttached(file)
    setAttached(img)
  }, [])

  const handlePaste = useCallback((e: React.ClipboardEvent) => {
    const items = Array.from(e.clipboardData?.items ?? [])
    const imageItem = items.find(item => item.type.startsWith('image/'))
    if (!imageItem) return
    const file = imageItem.getAsFile()
    if (file) {
      e.preventDefault()
      handleFile(file)
    }
  }, [handleFile])

  const submit = useCallback(() => {
    const el = textareaRef.current
    if (!el) return
    const value = el.value.trim()
    if ((!value && !attached) || disabled) return
    el.value = ''
    el.style.height = 'auto'
    onSend(value, attached ?? undefined)
    setAttached(null)
  }, [onSend, disabled, attached])

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
      {attached && (
        <div className="attachment-preview">
          <div className="attachment-card">
            <img className="attachment-thumb" src={attached.dataUrl} alt="" />
            <div className="attachment-info">
              <span className="attachment-name">{attached.name}</span>
              <span className="attachment-size">{formatSize(attached.size)}</span>
            </div>
            <button
              className="attachment-remove"
              onClick={() => setAttached(null)}
              title="Remove image"
              aria-label="Remove attached image"
            >
              <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <line x1="1" y1="1" x2="11" y2="11"/>
                <line x1="11" y1="1" x2="1" y2="11"/>
              </svg>
            </button>
          </div>
        </div>
      )}

      <div className="input-row">
        {/* Hidden file input */}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          style={{ display: 'none' }}
          onChange={e => {
            const file = e.target.files?.[0]
            if (file) handleFile(file)
            e.target.value = ''
          }}
        />

        {/* Upload button */}
        <button
          className="upload-btn"
          onClick={() => fileInputRef.current?.click()}
          disabled={disabled}
          title="Attach image"
          aria-label="Attach image"
        >
          <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/>
          </svg>
        </button>

        <textarea
          ref={textareaRef}
          className="user-input"
          placeholder="Message… (paste or attach an image)"
          rows={1}
          autoComplete="off"
          autoCorrect="off"
          spellCheck={false}
          onKeyDown={handleKeyDown}
          onInput={handleInput}
          onPaste={handlePaste}
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

      <div className="input-hint">Enter to send · Shift+Enter for new line · Paste or attach images</div>
    </div>
  )
}
