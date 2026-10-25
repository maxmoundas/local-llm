import { useCallback, useEffect, useState } from 'react'
import { api } from './api/client'
import type { AttachedImage, Chat, ModelOption } from './types'
import ChatView, { type PendingMessage } from './components/ChatView'
import { MenuIcon } from './components/Icons'
import NewChat from './components/NewChat'
import SettingsView from './components/SettingsView'
import Sidebar from './components/Sidebar'

const MODEL_KEY = 'modelId'
const SIDEBAR_KEY = 'sidebarOpen'

function load(key: string): string | null {
  try { return localStorage.getItem(key) } catch { return null }
}

function save(key: string, value: string): void {
  try { localStorage.setItem(key, value) } catch { /* storage unavailable */ }
}

const isNarrow = () => window.matchMedia('(max-width: 768px)').matches

export default function App() {
  const [chats, setChats] = useState<Chat[]>([])
  const [currentChatId, setCurrentChatId] = useState<string | null>(null)
  const [pending, setPending] = useState<{ chatId: string; message: PendingMessage } | null>(null)
  const [models, setModels] = useState<ModelOption[]>([])
  const [modelId, setModelId] = useState('')
  const [localModelLoaded, setLocalModelLoaded] = useState(false)
  const [view, setView] = useState<'chat' | 'settings'>('chat')
  const [sidebarOpen, setSidebarOpen] = useState(() => !isNarrow() && load(SIDEBAR_KEY) !== 'false')
  const [displayName, setDisplayName] = useState('')

  const refreshChats = useCallback(async () => {
    setChats(await api.listChats())
  }, [])

  useEffect(() => {
    api.listModels().then(ms => {
      setModels(ms)
      const saved = load(MODEL_KEY)
      // With no saved choice, default to a cloud model: usable immediately, while local takes minutes to load.
      const fallback = ms.find(m => m.provider !== 'local') ?? ms[0]
      setModelId(ms.some(m => m.id === saved) ? saved! : fallback?.id ?? '')
    })
    api.getSettings().then(s => setDisplayName(s.display_name ?? '')).catch(() => {})
    refreshChats()
  }, [refreshChats])

  useEffect(() => {
    if (localModelLoaded) return
    const check = () => api.health().then(h => { if (h.local_model_loaded) setLocalModelLoaded(true) }).catch(() => {})
    check()
    const id = setInterval(check, 3000)
    return () => clearInterval(id)
  }, [localModelLoaded])

  const chooseModel = useCallback((id: string) => {
    setModelId(id)
    save(MODEL_KEY, id)
  }, [])

  const toggleSidebar = useCallback(() => {
    setSidebarOpen(o => {
      save(SIDEBAR_KEY, String(!o))
      return !o
    })
  }, [])

  const closeSidebarOnMobile = () => { if (isNarrow()) setSidebarOpen(false) }

  const openChat = useCallback((id: string) => {
    setCurrentChatId(id)
    setView('chat')
    closeSidebarOnMobile()
  }, [])

  const newChat = useCallback(() => {
    setCurrentChatId(null)
    setView('chat')
    closeSidebarOnMobile()
  }, [])

  const startChat = useCallback(async (content: string, image?: AttachedImage) => {
    const chat = await api.createChat()
    setPending({ chatId: chat.id, message: { content, image } })
    setCurrentChatId(chat.id)
    refreshChats()
  }, [refreshChats])

  const deleteChat = useCallback(async (id: string) => {
    await api.deleteChat(id)
    if (currentChatId === id) setCurrentChatId(null)
    refreshChats()
  }, [currentChatId, refreshChats])

  const renameChat = useCallback(async (id: string, name: string) => {
    await api.renameChat(id, name)
    refreshChats()
  }, [refreshChats])

  const onTitle = useCallback((id: string, name: string) => {
    setChats(prev => prev.map(c => (c.id === id ? { ...c, name } : c)))
  }, [])

  const pinChat = useCallback(async (id: string) => {
    await api.pinChat(id)
    refreshChats()
  }, [refreshChats])

  const onChatLoad = useCallback((id: string | undefined) => {
    if (id && models.some(m => m.id === id)) setModelId(id)
  }, [models])

  const currentChat = chats.find(c => c.id === currentChatId)

  return (
    <div className="app">
      <Sidebar
        open={sidebarOpen}
        onToggle={toggleSidebar}
        chats={chats}
        currentChatId={currentChatId}
        onSelect={openChat}
        onNewChat={newChat}
        onDelete={deleteChat}
        onRename={renameChat}
        onPin={pinChat}
        onSettings={() => { setView(v => (v === 'settings' ? 'chat' : 'settings')); closeSidebarOnMobile() }}
        settingsActive={view === 'settings'}
        displayName={displayName}
      />
      <main className="main">
        <header className="topbar">
          {!sidebarOpen && (
            <button className="icon-btn mobile-only" onClick={toggleSidebar} aria-label="Open sidebar"><MenuIcon /></button>
          )}
          {view === 'chat' && currentChat && (
            <ChatTitle key={currentChat.id + currentChat.name} name={currentChat.name}
                       onRename={name => renameChat(currentChat.id, name)} />
          )}
        </header>

        {view === 'settings' ? (
          <SettingsView onDisplayName={setDisplayName} />
        ) : currentChatId ? (
          <ChatView
            key={currentChatId}
            chatId={currentChatId}
            initialMessage={pending?.chatId === currentChatId ? pending.message : undefined}
            models={models}
            modelId={modelId}
            onModelChange={chooseModel}
            onTitle={onTitle}
            onChatsChange={refreshChats}
            onChatLoad={onChatLoad}
            localModelLoaded={localModelLoaded}
          />
        ) : (
          <NewChat
            displayName={displayName}
            models={models}
            modelId={modelId}
            onModelChange={chooseModel}
            localModelLoaded={localModelLoaded}
            onSend={startChat}
          />
        )}
      </main>
    </div>
  )
}

function ChatTitle({ name, onRename }: { name: string; onRename: (name: string) => void }) {
  const [editing, setEditing] = useState(false)
  if (!editing) {
    return <button className="chat-title" onClick={() => setEditing(true)} title="Rename">{name}</button>
  }
  return (
    <input
      className="chat-title-input"
      defaultValue={name}
      autoFocus
      onFocus={e => e.currentTarget.select()}
      onBlur={e => {
        const v = e.currentTarget.value.trim()
        if (v && v !== name) onRename(v)
        setEditing(false)
      }}
      onKeyDown={e => {
        if (e.key === 'Enter') e.currentTarget.blur()
        if (e.key === 'Escape') setEditing(false)
      }}
    />
  )
}
