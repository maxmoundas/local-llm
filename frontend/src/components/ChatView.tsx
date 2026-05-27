import { useState, useEffect, useRef, useCallback } from 'react'
import { api } from '../api/client'
import type { Message, AttachedImage, ModelOption } from '../types'
import MessageBubble from './Message'
import InputBar from './InputBar'
import ModelSelector from './ModelSelector'

interface DisplayMessage extends Message {
  id: string
  streaming?: boolean
  imageDataUrl?: string
}

interface Props {
  chatId: string
  models: ModelOption[]
  selectedModelId: string
  onModelChange: (id: string) => void
  onRename: (chatId: string, name: string) => void
  onChatsChange: () => void
  onChatLoad: (modelId: string | undefined) => void
  localModelLoaded: boolean
}

export default function ChatView({ chatId, models, selectedModelId, onModelChange, onRename, onChatsChange, onChatLoad, localModelLoaded }: Props) {
  const [messages, setMessages] = useState<DisplayMessage[]>([])
  const [chatName, setChatName] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const [streamingContent, setStreamingContent] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const accumulatedRef = useRef('')

  useEffect(() => {
    api.getChat(chatId).then(chat => {
      setChatName(chat.name)
      setMessages(
        chat.messages
          .filter(m => m.role !== 'system')
          .map((m, i) => ({ ...m, id: String(i) }))
      )
      onChatLoad(chat.model_id)
    })
  }, [chatId])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages.length])

  const handleSend = useCallback(async (content: string, image?: AttachedImage) => {
    if (isGenerating) return
    setIsGenerating(true)
    accumulatedRef.current = ''
    setStreamingContent('')

    const userMsg: DisplayMessage = {
      id: `u-${Date.now()}`,
      role: 'user',
      content,
      imageDataUrl: image?.dataUrl,
    }
    const streamingMsg: DisplayMessage = {
      id: 'streaming', role: 'assistant', content: '', streaming: true,
    }
    setMessages(prev => [...prev, userMsg, streamingMsg])

    try {
      const res = await api.sendMessage(chatId, content, image?.base64, selectedModelId)
      if (!res.ok) throw new Error(`Server error ${res.status}`)

      const reader = res.body!.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let finalName = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const parts = buffer.split('\n\n')
        buffer = parts.pop()!

        for (const part of parts) {
          for (const line of part.split('\n')) {
            if (!line.startsWith('data: ')) continue
            const raw = line.slice(6).trim()
            if (!raw) continue
            let evt: { type: string; content?: string; name?: string; message?: string; target?: string; reason?: string }
            try {
              evt = JSON.parse(raw)
            } catch { continue /* malformed SSE line — skip */ }
            if (evt.type === 'token') {
              accumulatedRef.current += evt.content ?? ''
              setStreamingContent(accumulatedRef.current)
            } else if (evt.type === 'done') {
              finalName = evt.name ?? ''
            } else if (evt.type === 'error') {
              throw new Error(evt.message ?? 'Unknown error from server')
            } else if (evt.type === 'blocked') {
              const label = evt.target === 'input' ? 'Input' : 'Response'
              const suffix = evt.reason ? `: ${evt.reason}` : ''
              setMessages(prev => [
                ...prev.filter(m => m.id !== 'streaming'),
                { id: `blocked-${Date.now()}`, role: 'assistant', content: `🚫 ${label} blocked by safety guardrails${suffix}` },
              ])
              return  // finally block handles isGenerating + cleanup
            }
          }
        }
      }

      const finalContent = accumulatedRef.current
      setMessages(prev => [
        ...prev.filter(m => m.id !== 'streaming'),
        { id: `a-${Date.now()}`, role: 'assistant', content: finalContent },
      ])

      if (finalName && finalName !== chatName) {
        setChatName(finalName)
        onRename(chatId, finalName)
      }
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : 'Unknown error'
      setMessages(prev => [
        ...prev.filter(m => m.id !== 'streaming'),
        { id: `err-${Date.now()}`, role: 'assistant', content: `⚠ ${errMsg}` },
      ])
    } finally {
      setIsGenerating(false)
      accumulatedRef.current = ''
      setStreamingContent('')
      onChatsChange()
    }
  }, [chatId, chatName, isGenerating, selectedModelId, onRename, onChatsChange])

  const handleTitleChange = (e: React.FocusEvent<HTMLInputElement>) => {
    const name = e.target.value.trim() || chatName
    e.target.value = name
    if (name !== chatName) {
      setChatName(name)
      onRename(chatId, name)
    }
  }

  return (
    <div className="chat-view">
      <div className="chat-header">
        <input
          key={chatName}
          className="chat-title-input"
          defaultValue={chatName}
          onBlur={handleTitleChange}
          onKeyDown={e => { if (e.key === 'Enter') e.currentTarget.blur() }}
          title="Click to rename"
        />
        <ModelSelector
          models={models}
          selected={selectedModelId}
          onChange={onModelChange}
          localModelLoaded={localModelLoaded}
        />
      </div>

      <div className="messages">
        {messages.map(msg =>
          msg.id === 'streaming' ? (
            <MessageBubble
              key="streaming"
              role="assistant"
              content={streamingContent}
              streaming
            />
          ) : (
            <MessageBubble
              key={msg.id}
              role={msg.role as 'user' | 'assistant'}
              content={msg.content}
              imageDataUrl={msg.imageDataUrl}
            />
          )
        )}
        <div ref={messagesEndRef} />
      </div>

      <InputBar onSend={handleSend} disabled={isGenerating} />
    </div>
  )
}
