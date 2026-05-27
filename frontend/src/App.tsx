import { useState, useEffect, useCallback } from 'react'
import { api } from './api/client'
import type { Chat, ModelOption } from './types'
import Sidebar from './components/Sidebar'
import ChatView from './components/ChatView'
import EmptyState from './components/EmptyState'
import SettingsView from './components/SettingsView'

export default function App() {
  const [chats, setChats] = useState<Chat[]>([])
  const [currentChatId, setCurrentChatId] = useState<string | null>(null)
  const [models, setModels] = useState<ModelOption[]>([])
  const [selectedModelId, setSelectedModelId] = useState<string>('')
  const [localModelLoaded, setLocalModelLoaded] = useState(false)
  const [view, setView] = useState<'chat' | 'settings'>('chat')

  useEffect(() => {
    api.listModels().then(ms => {
      setModels(ms)
      if (ms.length > 0) setSelectedModelId(ms[0].id)
    })
    refreshChats()
  }, [])

  useEffect(() => {
    if (localModelLoaded) return
    const check = () => {
      api.health().then(h => {
        if (h.local_model_loaded) setLocalModelLoaded(true)
      }).catch(() => {})
    }
    check()
    const id = setInterval(check, 3000)
    return () => clearInterval(id)
  }, [localModelLoaded])

  const refreshChats = useCallback(async () => {
    setChats(await api.listChats())
  }, [])

  const createChat = useCallback(async () => {
    const chat = await api.createChat()
    await refreshChats()
    setCurrentChatId(chat.id)
  }, [refreshChats])

  const deleteChat = useCallback(async (chatId: string) => {
    await api.deleteChat(chatId)
    if (currentChatId === chatId) setCurrentChatId(null)
    await refreshChats()
  }, [currentChatId, refreshChats])

  const handleModelChange = useCallback(async (id: string) => {
    setSelectedModelId(id)
    const chat = await api.createChat()
    await refreshChats()
    setCurrentChatId(chat.id)
  }, [refreshChats])

  const handleRename = useCallback(async (chatId: string, name: string) => {
    await api.renameChat(chatId, name)
    await refreshChats()
  }, [refreshChats])

  const handleChatLoad = useCallback((modelId: string | undefined) => {
    if (modelId) setSelectedModelId(modelId)
  }, [])

  const handlePin = useCallback(async (chatId: string) => {
    await api.pinChat(chatId)
    await refreshChats()
  }, [refreshChats])

  const handleSelectChat = useCallback((id: string) => {
    setCurrentChatId(id)
    setView('chat')
  }, [])

  const handleCreateChat = useCallback(async () => {
    await createChat()
    setView('chat')
  }, [createChat])

  return (
    <div className="app">
      <Sidebar
        chats={chats}
        currentChatId={currentChatId}
        onSelect={handleSelectChat}
        onCreate={handleCreateChat}
        onDelete={deleteChat}
        onRename={handleRename}
        onPin={handlePin}
        onSettings={() => setView(v => v === 'settings' ? 'chat' : 'settings')}
        settingsActive={view === 'settings'}
      />
      <main className="main">
        {!localModelLoaded && (
          <div className="model-loading-banner">
            <span className="model-loading-dot" />
            Local model loading — Anthropic &amp; OpenAI models are available now
          </div>
        )}
        {view === 'settings' ? (
          <SettingsView />
        ) : currentChatId ? (
          <ChatView
            key={currentChatId}
            chatId={currentChatId}
            models={models}
            selectedModelId={selectedModelId}
            onModelChange={handleModelChange}
            onRename={handleRename}
            onChatsChange={refreshChats}
            onChatLoad={handleChatLoad}
            localModelLoaded={localModelLoaded}
          />
        ) : (
          <EmptyState onCreate={createChat} />
        )}
      </main>
    </div>
  )
}
