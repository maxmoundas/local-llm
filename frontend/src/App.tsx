import { useState, useEffect, useCallback } from 'react'
import { api } from './api/client'
import type { Chat } from './types'
import Sidebar from './components/Sidebar'
import ChatView from './components/ChatView'
import EmptyState from './components/EmptyState'

export default function App() {
  const [chats, setChats] = useState<Chat[]>([])
  const [currentChatId, setCurrentChatId] = useState<string | null>(null)
  const [modelName, setModelName] = useState('')

  useEffect(() => {
    api.health().then(h => setModelName(h.model?.split('/').pop() ?? 'model'))
    refreshChats()
  }, [])

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

  const handleRename = useCallback(async (chatId: string, name: string) => {
    await api.renameChat(chatId, name)
    await refreshChats()
  }, [refreshChats])

  return (
    <div className="app">
      <Sidebar
        chats={chats}
        currentChatId={currentChatId}
        onSelect={setCurrentChatId}
        onCreate={createChat}
        onDelete={deleteChat}
      />
      <main className="main">
        {currentChatId ? (
          <ChatView
            key={currentChatId}
            chatId={currentChatId}
            modelName={modelName}
            onRename={handleRename}
            onChatsChange={refreshChats}
          />
        ) : (
          <EmptyState onCreate={createChat} />
        )}
      </main>
    </div>
  )
}
