import { useCallback, useEffect, useRef, useState } from 'react'
import type { AttachedImage, ModelOption } from '../types'
import { ArrowUpIcon, CloseIcon, PaperclipIcon, StopIcon } from './Icons'
import ModelMenu from './ModelMenu'

interface Props {
  onSend: (content: string, image?: AttachedImage) => void
  onStop?: () => void
  isGenerating: boolean
  models: ModelOption[]
  modelId: string
  onModelChange: (id: string) => void
  localModelLoaded: boolean
  placeholder?: string
  menuPlacement?: 'up' | 'down'
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function fileToAttached(file: File): Promise<AttachedImage> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      const dataUrl = reader.result as string
      resolve({ dataUrl, base64: dataUrl.split(',')[1], name: file.name, size: file.size })
    }
    reader.onerror = reject
    reader.readAsDataURL(file)
  })
}

export default function Composer({
  onSend, onStop, isGenerating, models, modelId, onModelChange, localModelLoaded,
  placeholder = 'Reply…', menuPlacement = 'up',
}: Props) {
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [value, setValue] = useState('')
  const [attached, setAttached] = useState<AttachedImage | null>(null)

  const selected = models.find(m => m.id === modelId)
  const localLoading = selected?.provider === 'local' && !localModelLoaded
  const canSend = (value.trim().length > 0 || attached !== null) && !isGenerating && !localLoading

  useEffect(() => { textareaRef.current?.focus() }, [])

  // Grow with content up to ~40% of the viewport, then scroll.
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, window.innerHeight * 0.4)}px`
  }, [value])

  const handleFile = useCallback(async (file: File) => {
    if (!file.type.startsWith('image/')) return
    setAttached(await fileToAttached(file))
  }, [])

  const handlePaste = useCallback((e: React.ClipboardEvent) => {
    const item = Array.from(e.clipboardData?.items ?? []).find(i => i.type.startsWith('image/'))
    const file = item?.getAsFile()
    if (file) {
      e.preventDefault()
      handleFile(file)
    }
  }, [handleFile])

  const submit = () => {
    if (!canSend) return
    onSend(value.trim(), attached ?? undefined)
    setValue('')
    setAttached(null)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault()
      submit()
    }
  }

  return (
    <div className="composer-wrap">
      <div
        className="composer"
        onClick={e => { if (e.target === e.currentTarget) textareaRef.current?.focus() }}
        onDragOver={e => e.preventDefault()}
        onDrop={e => {
          e.preventDefault()
          const file = e.dataTransfer.files?.[0]
          if (file) handleFile(file)
        }}
      >
        {attached && (
          <div className="attachment">
            <img className="attachment-thumb" src={attached.dataUrl} alt="" />
            <div className="attachment-info">
              <span className="attachment-name">{attached.name}</span>
              <span className="attachment-size">{formatSize(attached.size)}</span>
            </div>
            <button className="attachment-remove" onClick={() => setAttached(null)} aria-label="Remove image">
              <CloseIcon size={12} />
            </button>
          </div>
        )}

        <textarea
          ref={textareaRef}
          className="composer-input"
          placeholder={placeholder}
          rows={1}
          value={value}
          onChange={e => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          onPaste={handlePaste}
        />

        <div className="composer-toolbar">
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            hidden
            onChange={e => {
              const file = e.target.files?.[0]
              if (file) handleFile(file)
              e.target.value = ''
            }}
          />
          <button className="icon-btn bordered" onClick={() => fileInputRef.current?.click()}
                  title="Attach image" aria-label="Attach image">
            <PaperclipIcon size={17} />
          </button>

          <div className="composer-toolbar-right">
            <ModelMenu models={models} selected={modelId} onChange={onModelChange}
                       localModelLoaded={localModelLoaded} placement={menuPlacement} />
            {isGenerating ? (
              <button className="send-btn stop" onClick={onStop} title="Stop" aria-label="Stop generating">
                <StopIcon size={16} />
              </button>
            ) : (
              <button className="send-btn" onClick={submit} disabled={!canSend} title="Send" aria-label="Send message">
                <ArrowUpIcon size={16} />
              </button>
            )}
          </div>
        </div>
      </div>
      {localLoading && (
        <div className="composer-note">The local model is still loading — pick a cloud model or wait a moment.</div>
      )}
    </div>
  )
}
