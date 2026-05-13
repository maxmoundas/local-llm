import { useState, useEffect, useRef, useCallback } from 'react'
import { api } from '../api/client'
import type { Message, AttachedImage } from '../types'
import MessageBubble from './Message'
import InputBar from './InputBar'

interface DisplayMessage extends Message {
  id: string
  streaming?: boolean
  imageDataUrl?: string
}

interface Props {
  chatId: string
  modelName: string
  onRename: (chatId: string, name: string) => void
  onChatsChange: () => void
}

export default function ChatView({ chatId, modelName, onRename, onChatsChange }: Props) {
  const [messages, setMessages] = useState<DisplayMessage[]>([])
  const [chatName, setChatName] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const streamingSpanRef = useRef<HTMLSpanElement>(null)
  const accumulatedRef = useRef('')

  useEffect(() => {
    api.getChat(chatId).then(chat => {
      setChatName(chat.name)
      setMessages(
        chat.messages
          .filter(m => m.role !== 'system')
          .map((m, i) => ({ ...m, id: String(i) }))
      )
    })
  }, [chatId])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages.length])

  const handleSend = useCallback(async (content: string, image?: AttachedImage) => {
    if (isGenerating) return
    setIsGenerating(true)
    accumulatedRef.current = ''

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
      const res = await api.sendMessage(chatId, content, image?.base64)
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
            try {
              const evt = JSON.parse(raw)
              if (evt.type === 'token') {
                accumulatedRef.current += evt.content
                if (streamingSpanRef.current) {
                  streamingSpanRef.current.textContent = accumulatedRef.current
                  messagesEndRef.current?.scrollIntoView({ behavior: 'instant' })
                }
              } else if (evt.type === 'done') {
                finalName = evt.name ?? ''
              } else if (evt.type === 'error') {
                throw new Error(evt.message)
              }
            } catch { /* malformed SSE line — skip */ }
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
      onChatsChange()
    }
  }, [chatId, chatName, isGenerating, onRename, onChatsChange])

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
        <span className="model-badge">{modelName}</span>
      </div>

      <div className="messages">
        {messages.map(msg =>
          msg.id === 'streaming' ? (
            <MessageBubble
              key="streaming"
              role="assistant"
              content=""
              streaming
              streamingRef={streamingSpanRef}
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
