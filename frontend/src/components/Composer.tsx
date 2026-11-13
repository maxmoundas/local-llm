import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from 'react'
import { api, formatSize, readFile } from '../api/client'
import type { ModelOption, PendingFile } from '../types'
import { ArrowUpIcon, CloseIcon, FileIcon, MicIcon, PaperclipIcon, StopIcon } from './Icons'
import ModelMenu from './ModelMenu'

interface Props {
  onSend: (content: string, files: PendingFile[]) => void
  onStop?: () => void
  isGenerating: boolean
  models: ModelOption[]
  modelId: string
  onModelChange: (id: string) => void
  localModelLoaded: boolean
  voiceAvailable?: boolean
  placeholder?: string
  menuPlacement?: 'up' | 'down'
}

export interface ComposerHandle {
  focus: () => void
}

const ACCEPT = 'image/png,image/jpeg,image/gif,image/webp,.pdf,.docx,.xlsx,.txt,.md,.csv,.tsv,.json,.jsonl,.yaml,.yml,'
  + '.xml,.html,.htm,.py,.js,.ts,.tsx,.jsx,.java,.go,.rs,.rb,.php,.c,.h,.cpp,.hpp,.cs,.swift,.kt,.sql,.sh,.toml,.ini,'
  + '.log,.tex,.css,.r,.lua'
const MAX_FILES = 10

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader()
    r.onload = () => resolve(String(r.result).split(',')[1] ?? '')
    r.onerror = reject
    r.readAsDataURL(blob)
  })
}

/** Record from the microphone, then transcribe with the backend's local Whisper. */
function useVoice(onText: (text: string) => void) {
  const [state, setState] = useState<'idle' | 'recording' | 'transcribing'>('idle')
  const [error, setError] = useState<string | null>(null)
  const recorder = useRef<MediaRecorder | null>(null)

  const start = async () => {
    setError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const rec = new MediaRecorder(stream)
      const chunks: Blob[] = []
      rec.ondataavailable = e => { if (e.data.size) chunks.push(e.data) }
      rec.onstop = async () => {
        stream.getTracks().forEach(t => t.stop())
        setState('transcribing')
        try {
          const blob = new Blob(chunks, { type: rec.mimeType || 'audio/webm' })
          const { text } = await api.transcribe(await blobToBase64(blob), blob.type)
          if (text) onText(text)
        } catch (err) {
          setError(err instanceof Error ? err.message : 'Transcription failed')
        }
        setState('idle')
      }
      rec.start()
      recorder.current = rec
      setState('recording')
    } catch {
      setError('Microphone unavailable — check the browser permission.')
    }
  }

  const stop = () => recorder.current?.state === 'recording' && recorder.current.stop()
  return { state, error, start, stop, toggle: () => (state === 'recording' ? stop() : state === 'idle' ? start() : undefined) }
}

const Composer = forwardRef<ComposerHandle, Props>(function Composer({
  onSend, onStop, isGenerating, models, modelId, onModelChange, localModelLoaded, voiceAvailable,
  placeholder = 'Reply…', menuPlacement = 'up',
}, ref) {
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [value, setValue] = useState('')
  const [files, setFiles] = useState<PendingFile[]>([])
  const [dragging, setDragging] = useState(false)

  useImperativeHandle(ref, () => ({ focus: () => textareaRef.current?.focus() }), [])

  const selected = models.find(m => m.id === modelId)
  const localLoading = selected?.provider === 'local' && !localModelLoaded
  const canSend = (value.trim().length > 0 || files.length > 0) && !isGenerating && !localLoading
  const hasImages = files.some(f => f.kind === 'image')
  const visionWarning = hasImages && selected && !selected.vision

  const voice = useVoice(text => {
    setValue(v => (v.trim() ? `${v.trimEnd()} ${text}` : text))
    textareaRef.current?.focus()
  })

  useEffect(() => { textareaRef.current?.focus() }, [])

  // Grow with content up to ~40% of the viewport, then scroll.
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, window.innerHeight * 0.4)}px`
  }, [value])

  const addFiles = useCallback(async (list: FileList | File[]) => {
    const picked = await Promise.all(Array.from(list).slice(0, MAX_FILES).map(readFile))
    setFiles(prev => [...prev, ...picked].slice(0, MAX_FILES))
  }, [])

  const handlePaste = useCallback((e: React.ClipboardEvent) => {
    const pasted = Array.from(e.clipboardData?.files ?? [])
    if (pasted.length) {
      e.preventDefault()
      addFiles(pasted)
    }
  }, [addFiles])

  const submit = () => {
    if (!canSend) return
    onSend(value.trim(), files)
    setValue('')
    setFiles([])
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
        className={`composer${dragging ? ' dragging' : ''}`}
        onClick={e => { if (e.target === e.currentTarget) textareaRef.current?.focus() }}
        onDragOver={e => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={e => {
          e.preventDefault()
          setDragging(false)
          if (e.dataTransfer.files?.length) addFiles(e.dataTransfer.files)
        }}
      >
        {files.length > 0 && (
          <div className="attachments">
            {files.map(f => (
              <div className="attachment" key={f.key}>
                {f.dataUrl
                  ? <img className="attachment-thumb" src={f.dataUrl} alt="" />
                  : <span className="attachment-thumb doc"><FileIcon size={18} /></span>}
                <div className="attachment-info">
                  <span className="attachment-name">{f.name}</span>
                  <span className="attachment-size">{formatSize(f.size)}</span>
                </div>
                <button className="attachment-remove" onClick={() => setFiles(prev => prev.filter(x => x.key !== f.key))}
                        aria-label={`Remove ${f.name}`}>
                  <CloseIcon size={12} />
                </button>
              </div>
            ))}
          </div>
        )}

        <textarea
          ref={textareaRef}
          className="composer-input"
          placeholder={voice.state === 'recording' ? 'Listening… click the mic to stop' : placeholder}
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
            accept={ACCEPT}
            multiple
            hidden
            onChange={e => {
              if (e.target.files?.length) addFiles(e.target.files)
              e.target.value = ''
            }}
          />
          <button className="icon-btn bordered" onClick={() => fileInputRef.current?.click()}
                  title="Attach files (images, PDF, Word, Excel, text, code)" aria-label="Attach files">
            <PaperclipIcon size={17} />
          </button>
          {voiceAvailable && (
            <button className={`icon-btn bordered mic-btn ${voice.state}`} onClick={voice.toggle}
                    disabled={voice.state === 'transcribing'}
                    title={voice.state === 'recording' ? 'Stop and transcribe' : 'Voice input (transcribed locally)'}
                    aria-label={voice.state === 'recording' ? 'Stop recording' : 'Start voice input'}>
              {voice.state === 'transcribing' ? <span className="spinner small" /> : <MicIcon size={17} />}
            </button>
          )}

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
      {visionWarning && (
        <div className="composer-note">{selected!.name} can't see images — pick a vision model, or the image will be skipped.</div>
      )}
      {voice.error && <div className="composer-note error">{voice.error}</div>}
    </div>
  )
})

export default Composer
