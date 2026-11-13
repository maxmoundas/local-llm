import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { api } from '../api/client'
import { readEvents } from '../lib/sse'
import { applyEvent, initialTurn, savedActions, type TurnState } from '../lib/turn'
import type { AttachedImage, Message, ModelOption } from '../types'
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

const MAX_RECONNECTS = 40
const sleep = (ms: number) => new Promise(r => setTimeout(r, ms))

function fromSaved(m: Message, i: number): DisplayMessage {
  return {
    id: String(i),
    role: m.role as 'user' | 'assistant',
    content: m.content,
    actions: savedActions(m),
    sources: m.sources,
    files: m.files,
    taskId: m.task_id,
    elapsedMs: m.elapsed_ms,
    usage: m.usage,
    stopped: m.stopped,
  }
}

function fromTurn(t: TurnState, id: string): DisplayMessage {
  return {
    id,
    role: 'assistant',
    content: t.content,
    actions: t.actions,
    sources: t.sources,
    files: t.files,
    taskId: t.taskId,
    status: t.status,
    pending: t.pending,
    startedAt: t.startedAt,
    elapsedMs: t.endedAt ? t.endedAt - t.startedAt : undefined,
    usage: t.usage,
    streaming: !t.ended,
    stopped: t.stopped,
    notice: t.notice,
  }
}

export default function ChatView({
  chatId, initialMessage, models, modelId, onModelChange, onTitle, onChatsChange, onChatLoad, localModelLoaded,
}: Props) {
  const [messages, setMessages] = useState<DisplayMessage[]>([])
  const [turn, setTurn] = useState<TurnState | null>(null)   // the live turn, rendered after `messages`
  const turnRef = useRef<TurnState | null>(null)
  const followingRef = useRef(false)
  const aliveRef = useRef(true)
  const scrollRef = useRef<HTMLDivElement>(null)
  const pinnedToBottom = useRef(true)
  const startedRef = useRef(false)  // a send began, so a late history load must not overwrite it

  useEffect(() => {
    aliveRef.current = true
    return () => { aliveRef.current = false }  // stop following; the task keeps running server-side
  }, [])

  /** Fold a task's events into the live turn until it ends, reconnecting when the stream drops. */
  const follow = useCallback(async (first: Response | null, start: TurnState) => {
    followingRef.current = true
    let state = start
    let res = first
    let failures = 0
    const update = (s: TurnState) => { state = s; turnRef.current = s; setTurn(s) }
    update(state)

    while (aliveRef.current && !state.ended) {
      try {
        if (!res) {
          if (!state.taskId) throw new Error('Lost the connection before the task started')
          res = await api.taskEvents(state.taskId, state.lastSeq)
        }
        if (!res.ok) {
          const detail = await res.json().catch(() => null)
          const err = new Error(detail?.detail ?? `Server error ${res.status}`)
          if (res.status < 500) { failures = MAX_RECONNECTS; throw err }  // not retryable
          throw err
        }
        for await (const evt of readEvents(res)) {
          if (!aliveRef.current) return
          update(applyEvent(state, evt))
          failures = 0
          if (evt.type === 'done' && evt.name) onTitle(chatId, evt.name)
        }
        res = null
      } catch (err) {
        res = null
        failures += 1
        if (!state.taskId || failures >= MAX_RECONNECTS) {
          const text = err instanceof Error ? err.message : 'Unknown error'
          update({ ...state, ended: true, endedAt: Date.now(), notice: state.notice ?? { kind: 'error', text } })
          break
        }
        await sleep(Math.min(400 * failures, 4000))
      }
    }
    followingRef.current = false
    if (!aliveRef.current) return
    const done = turnRef.current!
    setMessages(prev => [...prev, fromTurn(done, `a-${done.startedAt}`)])
    turnRef.current = null
    setTurn(null)
    onChatsChange()
  }, [chatId, onTitle, onChatsChange])

  // Load history, then reattach to a task that is still running (page reload, or came back to this chat).
  useEffect(() => {
    if (initialMessage) return
    let cancelled = false
    ;(async () => {
      const chat = await api.getChat(chatId)
      if (cancelled || startedRef.current) return
      setMessages(chat.messages.filter(m => m.role !== 'system').map(fromSaved))
      onChatLoad(chat.model_id)
      const active = await api.activeTask(chatId).catch(() => null)
      if (cancelled || startedRef.current || !active?.task_id) return
      startedRef.current = true
      follow(null, { ...initialTurn(), taskId: active.task_id })
    })()
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chatId])

  // Follow the stream only while the user is at the bottom; scrolling up pauses it.
  useLayoutEffect(() => {
    const el = scrollRef.current
    if (el && pinnedToBottom.current) el.scrollTop = el.scrollHeight
  }, [messages, turn])

  const onScroll = () => {
    const el = scrollRef.current
    if (el) pinnedToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80
  }

  const send = useCallback(async (content: string, image?: AttachedImage) => {
    if (followingRef.current) return
    startedRef.current = true
    pinnedToBottom.current = true
    setMessages(prev => [...prev, { id: `u-${Date.now()}`, role: 'user', content, imageDataUrl: image?.dataUrl }])
    const start = initialTurn()
    try {
      const res = await api.sendMessage(chatId, content, image?.base64, modelId)
      const taskId = res.headers.get('X-Task-Id') ?? undefined
      await follow(res, { ...start, taskId })
    } catch (err) {
      const text = err instanceof Error ? err.message : 'Unknown error'
      setMessages(prev => [...prev, fromTurn({ ...start, ended: true, notice: { kind: 'error', text } }, `a-${Date.now()}`)])
    }
  }, [chatId, modelId, follow])

  // Stop = cancel the task. The stream keeps flowing until the server confirms with task_end,
  // so whatever was written so far is shown and saved.
  const stop = useCallback(() => {
    const id = turnRef.current?.taskId
    if (id) api.cancelTask(id).catch(() => {})
  }, [])

  const reply = useCallback((r: { request_id: string; answer?: string; approved?: boolean }) => {
    const id = turnRef.current?.taskId
    if (id) api.replyTask(id, r).catch(() => {})
  }, [])

  // First message typed on the new-chat screen.
  useEffect(() => {
    if (initialMessage && !startedRef.current) send(initialMessage.content, initialMessage.image)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // The live turn and the message it becomes share a key, so the view (and its open/closed
  // state) carries over without a remount when the task ends.
  const all = turn ? [...messages, fromTurn(turn, `a-${turn.startedAt}`)] : messages
  return (
    <div className="chat-view">
      <div className="messages-scroll" ref={scrollRef} onScroll={onScroll}>
        <div className="messages">
          {all.map((msg, i) => (
            <MessageView key={msg.id} msg={msg} isLast={i === all.length - 1}
                         onReply={turn && i === all.length - 1 ? reply : undefined} />
          ))}
        </div>
      </div>
      <div className="composer-dock">
        <Composer
          onSend={send}
          onStop={stop}
          isGenerating={!!turn && !turn.ended}
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
