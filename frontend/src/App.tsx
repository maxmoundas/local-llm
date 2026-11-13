import { useCallback, useEffect, useState } from 'react'
import { api, AUTH_EVENT } from './api/client'
import { PreviewContext } from './lib/preview'
import type { Chat, ModelOption, PendingFile, Preview, Project } from './types'
import ChatSettingsMenu from './components/ChatSettingsMenu'
import ChatView, { type PendingMessage } from './components/ChatView'
import { LoginScreen, NewProjectDialog, ShortcutsDialog } from './components/Dialogs'
import { FolderIcon, KeyboardIcon, MenuIcon } from './components/Icons'
import NewChat from './components/NewChat'
import PreviewPanel from './components/PreviewPanel'
import ProjectView from './components/ProjectView'
import SchedulesView from './components/SchedulesView'
import SettingsView from './components/SettingsView'
import Sidebar, { type View } from './components/Sidebar'

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
  const [auth, setAuth] = useState<'checking' | 'ok' | 'login'>('checking')

  useEffect(() => {
    api.authStatus()
      .then(s => setAuth(s.required && !s.authenticated ? 'login' : 'ok'))
      .catch(() => setAuth('ok'))  // backend down: let the app show its own errors
    const onAuth = () => setAuth('login')
    window.addEventListener(AUTH_EVENT, onAuth)
    return () => window.removeEventListener(AUTH_EVENT, onAuth)
  }, [])

  if (auth === 'checking') return null
  if (auth === 'login') return <LoginScreen onDone={() => setAuth('ok')} />
  return <Shell />
}

function Shell() {
  const [chats, setChats] = useState<Chat[]>([])
  const [projects, setProjects] = useState<Project[]>([])
  const [currentChatId, setCurrentChatId] = useState<string | null>(null)
  const [focusIndex, setFocusIndex] = useState<number | null>(null)
  const [pending, setPending] = useState<{ chatId: string; message: PendingMessage } | null>(null)
  const [models, setModels] = useState<ModelOption[]>([])
  const [modelId, setModelId] = useState('')
  const [localModelLoaded, setLocalModelLoaded] = useState(false)
  const [view, setView] = useState<View>({ kind: 'chat' })
  const [sidebarOpen, setSidebarOpen] = useState(() => !isNarrow() && load(SIDEBAR_KEY) !== 'false')
  const [displayName, setDisplayName] = useState('')
  const [voiceAvailable, setVoiceAvailable] = useState(false)
  const [preview, setPreview] = useState<Preview | null>(null)
  const [dialog, setDialog] = useState<'shortcuts' | 'new-project' | null>(null)
  const [searchSignal, setSearchSignal] = useState(0)

  const refreshChats = useCallback(async () => {
    setChats(await api.listChats())
  }, [])

  const refreshProjects = useCallback(async () => {
    setProjects(await api.listProjects())
  }, [])

  useEffect(() => {
    api.listModels().then(ms => {
      setModels(ms)
      const saved = load(MODEL_KEY)
      // With no saved choice, default to a cloud model: usable immediately, while local takes minutes to load.
      const fallback = ms.find(m => m.provider !== 'local') ?? ms[0]
      setModelId(ms.some(m => m.id === saved) ? saved! : fallback?.id ?? '')
    })
    api.getSettings().then(s => {
      setDisplayName(s.display_name ?? '')
      setVoiceAvailable(!!s.transcription_available)
    }).catch(() => {})
    refreshChats()
    refreshProjects()
  }, [refreshChats, refreshProjects])

  // Scheduled runs add messages in the background; keep the sidebar fresh.
  useEffect(() => {
    const id = setInterval(refreshChats, 60000)
    return () => clearInterval(id)
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

  const openChat = useCallback((id: string, index: number | null = null) => {
    setCurrentChatId(id)
    setFocusIndex(index)
    setView({ kind: 'chat' })
    closeSidebarOnMobile()
  }, [])

  const newChat = useCallback(() => {
    setCurrentChatId(null)
    setView({ kind: 'chat' })
    closeSidebarOnMobile()
  }, [])

  const startChat = useCallback(async (content: string, files: PendingFile[], projectId?: string) => {
    const chat = await api.createChat(projectId)
    setPending({ chatId: chat.id, message: { content, files } })
    setCurrentChatId(chat.id)
    setFocusIndex(null)
    setView({ kind: 'chat' })
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

  const generateTitle = useCallback(async (id: string) => {
    await api.generateTitle(id)
    refreshChats()
  }, [refreshChats])

  const moveChat = useCallback(async (id: string, projectId: string | null) => {
    await api.moveChat(id, projectId)
    refreshChats()
    refreshProjects()
  }, [refreshChats, refreshProjects])

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

  const createProject = useCallback(async (name: string, description: string) => {
    const p = await api.createProject({ name, description })
    await refreshProjects()
    setDialog(null)
    setView({ kind: 'project', id: p.id })
    closeSidebarOnMobile()
  }, [refreshProjects])

  // ── Keyboard shortcuts ──
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const mod = e.metaKey || e.ctrlKey
      const key = e.key.toLowerCase()
      if (mod && e.shiftKey && key === 'o') { e.preventDefault(); newChat() }
      else if (mod && !e.shiftKey && key === 'k') {
        e.preventDefault()
        setSidebarOpen(true)
        setSearchSignal(n => n + 1)
      } else if (mod && !e.shiftKey && key === 'b') { e.preventDefault(); toggleSidebar() }
      else if (mod && key === ',') { e.preventDefault(); setView({ kind: 'settings' }) }
      else if (mod && (key === '/' || key === '?')) { e.preventDefault(); setDialog(d => (d === 'shortcuts' ? null : 'shortcuts')) }
      else if (mod && e.shiftKey && key === 'c') {
        const replies = document.querySelectorAll<HTMLElement>('.msg.assistant .prose')
        const last = replies[replies.length - 1]
        if (last) { e.preventDefault(); navigator.clipboard.writeText(last.innerText) }
      } else if (e.shiftKey && e.key === 'Escape') {
        e.preventDefault()
        document.querySelector<HTMLTextAreaElement>('.composer-input')?.focus()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [newChat, toggleSidebar])

  const currentChat = chats.find(c => c.id === currentChatId)
  const currentProject = currentChat?.project_id ? projects.find(p => p.id === currentChat.project_id) : undefined
  const selectedModel = models.find(m => m.id === modelId)

  let content
  if (view.kind === 'settings') {
    content = <SettingsView onDisplayName={setDisplayName} />
  } else if (view.kind === 'schedules') {
    content = <SchedulesView models={models} projects={projects} defaultModelId={modelId}
                             onOpenChat={id => openChat(id)} onChatsChange={refreshChats} />
  } else if (view.kind === 'project') {
    content = (
      <ProjectView key={view.id} projectId={view.id} models={models} modelId={modelId} onModelChange={chooseModel}
                   localModelLoaded={localModelLoaded} voiceAvailable={voiceAvailable}
                   onStartChat={(pid, text, files) => startChat(text, files, pid)}
                   onOpenChat={id => openChat(id)} onChanged={refreshProjects}
                   onDeleted={() => { refreshProjects(); refreshChats(); newChat() }} />
    )
  } else if (currentChatId) {
    content = (
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
        voiceAvailable={voiceAvailable}
        focusIndex={focusIndex}
      />
    )
  } else {
    content = (
      <NewChat displayName={displayName} models={models} modelId={modelId} onModelChange={chooseModel}
               localModelLoaded={localModelLoaded} voiceAvailable={voiceAvailable} onSend={(t, f) => startChat(t, f)} />
    )
  }

  return (
    <PreviewContext.Provider value={setPreview}>
      <div className="app">
        <Sidebar
          open={sidebarOpen}
          onToggle={toggleSidebar}
          chats={chats}
          projects={projects}
          currentChatId={currentChatId}
          view={view}
          onSelect={openChat}
          onNewChat={newChat}
          onDelete={deleteChat}
          onRename={renameChat}
          onPin={pinChat}
          onGenerateTitle={generateTitle}
          onMove={moveChat}
          onOpenProject={id => { setView({ kind: 'project', id }); closeSidebarOnMobile() }}
          onNewProject={() => setDialog('new-project')}
          onSchedules={() => { setView({ kind: 'schedules' }); closeSidebarOnMobile() }}
          onSettings={() => { setView(v => (v.kind === 'settings' ? { kind: 'chat' } : { kind: 'settings' })); closeSidebarOnMobile() }}
          displayName={displayName}
          searchSignal={searchSignal}
        />
        <main className="main">
          <header className="topbar">
            {!sidebarOpen && (
              <button className="icon-btn mobile-only" onClick={toggleSidebar} aria-label="Open sidebar"><MenuIcon /></button>
            )}
            {view.kind === 'chat' && currentProject && (
              <button className="topbar-project" onClick={() => setView({ kind: 'project', id: currentProject.id })}>
                <FolderIcon size={14} /> {currentProject.name} <span className="topbar-sep">/</span>
              </button>
            )}
            {view.kind === 'chat' && currentChat && (
              <ChatTitle key={currentChat.id + currentChat.name} name={currentChat.name}
                         onRename={name => renameChat(currentChat.id, name)} />
            )}
            <div className="topbar-right">
              {view.kind === 'chat' && currentChatId && (
                <ChatSettingsMenu key={currentChatId} chatId={currentChatId} model={selectedModel} />
              )}
              <button className="icon-btn desktop-only" onClick={() => setDialog('shortcuts')} title="Keyboard shortcuts"
                      aria-label="Keyboard shortcuts"><KeyboardIcon size={17} /></button>
            </div>
          </header>
          {content}
        </main>
        {preview && <PreviewPanel preview={preview} onClose={() => setPreview(null)} />}
      </div>
      {dialog === 'shortcuts' && <ShortcutsDialog onClose={() => setDialog(null)} />}
      {dialog === 'new-project' && <NewProjectDialog onClose={() => setDialog(null)} onCreate={createProject} />}
    </PreviewContext.Provider>
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
