import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { api } from '../api/client'
import { readEvents } from '../lib/sse'
import type { AttachedImage, ModelOption } from '../types'
import Composer from './Composer'
import MessageView, { type DisplayMessage } from './Message'

export interface PendingMessage {
  content: string
  image?: AttachedImage
}

interface Props {
  chatId: string
  initialMessage?: PendingMessage   // sent once on mount (first message from the new-chat screen)
  models: ModelOption[]
  modelId: string
  onModelChange: (id: string) => void
  onTitle: (chatId: string, name: string) => void
  onChatsChange: () => void
  onChatLoad: (modelId: string | undefined) => void
  localModelLoaded: boolean
}

const PENDING_ID = 'pending'

export default function ChatView({
  chatId, initialMessage, models, modelId, onModelChange, onTitle, onChatsChange, onChatLoad, localModelLoaded,
}: Props) {
  const [messages, setMessages] = useState<DisplayMessage[]>([])
  const [isGenerating, setIsGenerating] = useState(false)
  const abortRef = useRef<AbortController | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const pinnedToBottom = useRef(true)
  const startedRef = useRef(false)  // a send began, so a late history load must not overwrite it

  useEffect(() => {
    if (initialMessage) return
    api.getChat(chatId).then(chat => {
      if (startedRef.current) return
      setMessages(
        chat.messages
          .filter(m => m.role !== 'system')
          .map((m, i) => ({
            id: String(i),
            role: m.role as 'user' | 'assistant',
            content: m.content,
            steps: m.steps,
            sources: m.sources,
            stopped: m.stopped,
          })),
      )
      onChatLoad(chat.model_id)
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chatId])

  // Leaving the chat does not abort: like Claude, the reply keeps generating server-side
  // and is saved, and the title/sidebar still update when it finishes.

  // Follow the stream only while the user is at the bottom; scrolling up pauses it.
  useLayoutEffect(() => {
    const el = scrollRef.current
    if (el && pinnedToBottom.current) el.scrollTop = el.scrollHeight
  }, [messages])

  const onScroll = () => {
    const el = scrollRef.current
    if (el) pinnedToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80
  }

  const patchPending = (fn: (m: DisplayMessage) => DisplayMessage) =>
    setMessages(prev => prev.map(m => (m.id === PENDING_ID ? fn(m) : m)))

  const finalizePending = (fn: (m: DisplayMessage) => DisplayMessage = m => m) =>
    setMessages(prev => prev.map(m => (m.id === PENDING_ID
      ? { ...fn(m), id: `a-${Date.now()}`, streaming: false, status: undefined }
      : m)))

  const send = useCallback(async (content: string, image?: AttachedImage) => {
    if (abortRef.current) return
    startedRef.current = true
    pinnedToBottom.current = true
    const controller = new AbortController()
    abortRef.current = controller
    setIsGenerating(true)

    setMessages(prev => [
      ...prev,
      { id: `u-${Date.now()}`, role: 'user', content, imageDataUrl: image?.dataUrl },
      { id: PENDING_ID, role: 'assistant', content: '', streaming: true, steps: [], sources: [] },
    ])

    try {
      const res = await api.sendMessage(chatId, content, image?.base64, modelId, controller.signal)
      if (!res.ok) {
        const detail = await res.json().catch(() => null)
        throw new Error(detail?.detail ?? `Server error ${res.status}`)
      }

      for await (const evt of readEvents(res)) {
        switch (evt.type) {
          case 'status':
            patchPending(m => ({ ...m, status: evt.text }))
            break
          case 'step':
            patchPending(m => ({ ...m, steps: [...(m.steps ?? []), evt.step] }))
            break
          case 'sources':
            patchPending(m => ({ ...m, sources: evt.sources }))
            break
          case 'token':
            patchPending(m => ({ ...m, content: m.content + evt.content }))
            break
          case 'blocked': {
            const label = evt.target === 'input' ? 'This message was' : 'This response was'
            const text = `${label} blocked by safety guardrails${evt.reason ? `: ${evt.reason}` : '.'}`
            finalizePending(m => ({ ...m, content: evt.target === 'output' ? '' : m.content, notice: { kind: 'blocked', text } }))
            return
          }
          case 'error':
            throw new Error(evt.message)
          case 'done':
            if (evt.name) onTitle(chatId, evt.name)
            break
        }
      }
      finalizePending()
    } catch (err) {
      if (controller.signal.aborted) {
        finalizePending(m => ({ ...m, stopped: true }))
      } else {
        const text = err instanceof Error ? err.message : 'Unknown error'
        finalizePending(m => ({ ...m, notice: { kind: 'error', text } }))
      }
    } finally {
      abortRef.current = null
      setIsGenerating(false)
      onChatsChange()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chatId, modelId, onTitle, onChatsChange])

  const stop = useCallback(() => abortRef.current?.abort(), [])

  // First message typed on the new-chat screen.
  useEffect(() => {
    if (initialMessage && !startedRef.current) send(initialMessage.content, initialMessage.image)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return (
    <div className="chat-view">
      <div className="messages-scroll" ref={scrollRef} onScroll={onScroll}>
        <div className="messages">
          {messages.map((msg, i) => (
            <MessageView key={msg.id} msg={msg} isLast={i === messages.length - 1} />
          ))}
        </div>
      </div>
      <div className="composer-dock">
        <Composer
          onSend={send}
          onStop={stop}
          isGenerating={isGenerating}
          models={models}
          modelId={modelId}
          onModelChange={onModelChange}
          localModelLoaded={localModelLoaded}
        />
        <div className="composer-disclaimer">AI can make mistakes. Check important info.</div>
      </div>
    </div>
  )
}
