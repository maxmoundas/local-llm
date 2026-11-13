import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api/client'
import { readEvents } from '../lib/sse'
import { applyEvent, initialTurn, savedActions, type TurnState } from '../lib/turn'
import type { ChatDetail, Message, ModelOption, PendingFile } from '../types'
import Composer from './Composer'
import MessageView, { type DisplayMessage, type MessageHandlers } from './Message'

export interface PendingMessage {
  content: string
  files: PendingFile[]
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
  voiceAvailable: boolean
  focusIndex?: number | null        // scroll to this message (from search)
}

const MAX_RECONNECTS = 40
const sleep = (ms: number) => new Promise(r => setTimeout(r, ms))

export function fromSaved(m: Message, i: number): DisplayMessage {
  return {
    id: `m-${i}`,
    index: i,
    role: m.role as 'user' | 'assistant',
    content: m.content,
    attachments: m.attachments,
    versionCount: m.versions ? m.versions.length + 1 : undefined,
    versionIndex: m.version_index,
    modelId: m.model_id,
    actions: savedActions(m),
    sources: m.sources,
    files: m.files,
    taskId: m.task_id,
    elapsedMs: m.elapsed_ms,
    usage: m.usage,
    stopped: m.stopped,
    memory: m.memory,
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
    memory: t.memory,
  }
}

/** Saved messages, plus the finished live turn if the server didn't save it (error / blocked). */
function rebuild(chat: ChatDetail, done: TurnState | null): DisplayMessage[] {
  const saved = chat.messages.filter(m => m.role !== 'system')
  const list = saved.map(fromSaved)
  if (!done) return list
  const liveId = `a-${done.startedAt}`
  const last = saved[saved.length - 1]
  if (last?.role === 'assistant' && last.task_id && last.task_id === done.taskId) {
    // Same key as the live view, so the message (and its open activity card) doesn't remount.
    list[list.length - 1] = { ...list[list.length - 1], id: liveId, notice: done.notice }
  } else if (done.notice || done.content) {
    list.push({ ...fromTurn(done, liveId), retryable: last?.role === 'user' && done.notice?.kind === 'error' })
  }
  return list
}

export default function ChatView({
  chatId, initialMessage, models, modelId, onModelChange, onTitle, onChatsChange, onChatLoad, localModelLoaded,
  voiceAvailable, focusIndex,
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
          if (res.status < 500) failures = MAX_RECONNECTS  // not retryable
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
    // Reload so indexes and versions match the server (regenerate / edit reshape the history).
    const chat = await api.getChat(chatId).catch(() => null)
    if (!aliveRef.current) return
    if (chat) setMessages(rebuild(chat, done))
    else setMessages(prev => [...prev, fromTurn(done, `a-${done.startedAt}`)])
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
      setMessages(rebuild(chat, null))
      onChatLoad(chat.model_id)
      const active = await api.activeTask(chatId).catch(() => null)
      if (cancelled || startedRef.current || !active?.task_id) return
      startedRef.current = true
      follow(null, { ...initialTurn(), taskId: active.task_id })
    })()
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chatId])

  // Jump to a search hit once its message is rendered.
  useEffect(() => {
    if (focusIndex == null || !messages.length) return
    const el = scrollRef.current?.querySelector(`[data-index="${focusIndex}"]`)
    if (el) {
      pinnedToBottom.current = false
      el.scrollIntoView({ block: 'center' })
      el.classList.add('flash')
      setTimeout(() => el.classList.remove('flash'), 1600)
    }
  }, [focusIndex, messages.length])

  // Follow the stream only while the user is at the bottom; scrolling up pauses it.
  useLayoutEffect(() => {
    const el = scrollRef.current
    if (el && pinnedToBottom.current) el.scrollTop = el.scrollHeight
  }, [messages, turn])

  const onScroll = () => {
    const el = scrollRef.current
    if (el) pinnedToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80
  }

  /** Start a turn: show `shown` right away, then follow the task the request starts. */
  const runTurn = useCallback(async (shown: (prev: DisplayMessage[]) => DisplayMessage[], request: () => Promise<Response>) => {
    if (followingRef.current) return
    followingRef.current = true
    startedRef.current = true
    pinnedToBottom.current = true
    setMessages(shown)
    const start = initialTurn()
    turnRef.current = start
    setTurn(start)
    try {
      const res = await request()
      const taskId = res.headers.get('X-Task-Id') ?? undefined
      await follow(res, { ...start, taskId })
    } catch (err) {
      const text = err instanceof Error ? err.message : 'Unknown error'
      followingRef.current = false
      turnRef.current = null
      setTurn(null)
      setMessages(prev => [...prev, fromTurn({ ...start, ended: true, notice: { kind: 'error', text } }, `a-${Date.now()}`)])
    }
  }, [follow])

  const send = useCallback((content: string, files: PendingFile[]) => {
    const optimistic: DisplayMessage = {
      id: `u-${Date.now()}`, role: 'user', content,
      localImages: files.filter(f => f.dataUrl).map(f => f.dataUrl!),
      attachments: files.filter(f => f.kind === 'document')
        .map(f => ({ id: f.key, name: f.name, mime: f.mime, size: f.size, kind: 'document' as const })),
    }
    runTurn(prev => [...prev.filter(m => m.index !== undefined), optimistic], async () => {
      const uploaded = await Promise.all(files.map(f => api.uploadFile(chatId, f)))
      return api.sendMessage(chatId, content, uploaded.map(a => a.id), modelId)
    })
  }, [chatId, modelId, runTurn])

  const edit = useCallback((index: number, content: string) => {
    runTurn(prev => {
      const old = prev.find(m => m.index === index)
      return [...prev.filter(m => m.index !== undefined && m.index < index),
              { ...(old ?? { role: 'user' as const }), id: `u-${Date.now()}`, content, index: undefined, versionCount: undefined }]
    }, () => api.editMessage(chatId, index, content, modelId))
  }, [chatId, modelId, runTurn])

  const regenerate = useCallback((index: number | null) => {
    runTurn(prev => prev.filter(m => m.index !== undefined && (index === null || m.index < index)),
            () => api.regenerate(chatId, index, modelId))
  }, [chatId, modelId, runTurn])

  const switchVersion = useCallback(async (index: number, version: number) => {
    if (followingRef.current) return
    const chat = await api.switchVersion(chatId, index, version)
    setMessages(rebuild(chat, null))
  }, [chatId])

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
    if (initialMessage && !startedRef.current) send(initialMessage.content, initialMessage.files)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const busy = !!turn && !turn.ended
  const handlers = useMemo<MessageHandlers>(() => ({
    onEdit: edit, onRegenerate: regenerate, onSwitchVersion: switchVersion,
  }), [edit, regenerate, switchVersion])
  const liveHandlers = useMemo<MessageHandlers>(() => ({ ...handlers, onReply: reply }), [handlers, reply])

  // The live turn and the message it becomes share a key, so the view (and its open/closed
  // state) carries over without a remount when the task ends.
  const all = turn ? [...messages, fromTurn(turn, `a-${turn.startedAt}`)] : messages
  return (
    <div className="chat-view">
      <div className="messages-scroll" ref={scrollRef} onScroll={onScroll}>
        <div className="messages">
          {all.map((msg, i) => (
            <div key={msg.id} data-index={msg.index}>
              <MessageView chatId={chatId} msg={msg} isLast={i === all.length - 1} busy={busy}
                           handlers={turn && i === all.length - 1 ? liveHandlers : handlers} />
            </div>
          ))}
        </div>
      </div>
      <div className="composer-dock">
        <Composer
          onSend={send}
          onStop={stop}
          isGenerating={busy}
          models={models}
          modelId={modelId}
          onModelChange={onModelChange}
          localModelLoaded={localModelLoaded}
          voiceAvailable={voiceAvailable}
        />
        <div className="composer-disclaimer">AI can make mistakes. Check important info.</div>
      </div>
    </div>
  )
}
