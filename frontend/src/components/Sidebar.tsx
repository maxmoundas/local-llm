import { useEffect, useRef, useState } from 'react'
import { api } from '../api/client'
import type { Chat, Project, SearchResult } from '../types'
import {
  ClockIcon, CloseIcon, DotsIcon, DownloadIcon, FolderIcon, PanelIcon, PencilIcon, PlusIcon, SearchIcon,
  SettingsIcon, SparkIcon, StarIcon, TrashIcon,
} from './Icons'

export type View =
  | { kind: 'chat' }
  | { kind: 'settings' }
  | { kind: 'schedules' }
  | { kind: 'project'; id: string }

interface Props {
  open: boolean
  onToggle: () => void
  chats: Chat[]
  projects: Project[]
  currentChatId: string | null
  view: View
  onSelect: (id: string, focusIndex?: number | null) => void
  onNewChat: () => void
  onDelete: (id: string) => void
  onRename: (id: string, name: string) => void
  onPin: (id: string) => void
  onGenerateTitle: (id: string) => void
  onMove: (id: string, projectId: string | null) => void
  onOpenProject: (id: string) => void
  onNewProject: () => void
  onSchedules: () => void
  onSettings: () => void
  displayName: string
  searchSignal: number     // bumped by the Cmd/Ctrl+K shortcut
}

const PROJECTS_SHOWN = 5

function ChatRow({ chat, active, projects, onSelect, onDelete, onRename, onPin, onGenerateTitle, onMove }: {
  chat: Chat
  active: boolean
  projects: Project[]
  onSelect: () => void
  onDelete: () => void
  onRename: (name: string) => void
  onPin: () => void
  onGenerateTitle: () => void
  onMove: (projectId: string | null) => void
}) {
  const [menuOpen, setMenuOpen] = useState(false)
  const [moving, setMoving] = useState(false)
  const [editing, setEditing] = useState(false)
  const [name, setName] = useState(chat.name)
  const menuRef = useRef<HTMLDivElement>(null)

  useEffect(() => setName(chat.name), [chat.name])

  useEffect(() => {
    if (!menuOpen) return
    const onDown = (e: MouseEvent) => { if (!menuRef.current?.contains(e.target as Node)) { setMenuOpen(false); setMoving(false) } }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [menuOpen])

  const commit = () => {
    const trimmed = name.trim()
    if (trimmed && trimmed !== chat.name) onRename(trimmed)
    else setName(chat.name)
    setEditing(false)
  }

  if (editing) {
    return (
      <div className="chat-row editing">
        <input
          className="chat-row-input"
          value={name}
          autoFocus
          onFocus={e => e.currentTarget.select()}
          onChange={e => setName(e.target.value)}
          onBlur={commit}
          onKeyDown={e => {
            if (e.key === 'Enter') e.currentTarget.blur()
            if (e.key === 'Escape') { setName(chat.name); setEditing(false) }
          }}
        />
      </div>
    )
  }

  const close = () => { setMenuOpen(false); setMoving(false) }
  const project = projects.find(p => p.id === chat.project_id)

  return (
    <div className={`chat-row${active ? ' active' : ''}${menuOpen ? ' menu-open' : ''}`} onClick={onSelect}>
      {chat.schedule_id && <ClockIcon size={14} className="chat-row-badge" aria-label="Scheduled" />}
      <span className="chat-row-name" onDoubleClick={() => setEditing(true)} title={project ? `${chat.name} · ${project.name}` : chat.name}>
        {chat.name}
      </span>
      <div className="chat-row-menu" ref={menuRef} onClick={e => e.stopPropagation()}>
        <button className="icon-btn small" onClick={() => setMenuOpen(o => !o)} aria-label="Chat options">
          <DotsIcon size={16} />
        </button>
        {menuOpen && (
          <div className="popover row-menu">
            {moving ? (<>
              <div className="row-menu-label">Move to project</div>
              {projects.map(p => (
                <button key={p.id} disabled={p.id === chat.project_id} onClick={() => { close(); onMove(p.id) }}>
                  <FolderIcon size={15} /> {p.name}
                </button>
              ))}
              {chat.project_id && (
                <button onClick={() => { close(); onMove(null) }}><CloseIcon size={15} /> Remove from project</button>
              )}
              {projects.length === 0 && <div className="row-menu-empty">No projects yet</div>}
            </>) : (<>
              <button onClick={() => { close(); onPin() }}>
                <StarIcon size={15} filled={chat.pinned} /> {chat.pinned ? 'Unstar' : 'Star'}
              </button>
              <button onClick={() => { close(); setEditing(true) }}>
                <PencilIcon size={15} /> Rename
              </button>
              <button onClick={() => { close(); onGenerateTitle() }}>
                <SparkIcon size={15} /> Generate title
              </button>
              <button onClick={() => setMoving(true)}>
                <FolderIcon size={15} /> {chat.project_id ? 'Move…' : 'Add to project…'}
              </button>
              <a href={api.exportUrl(chat.id, 'md')} download onClick={close}><DownloadIcon size={15} /> Export Markdown</a>
              <a href={api.exportUrl(chat.id, 'json')} download onClick={close}><DownloadIcon size={15} /> Export JSON</a>
              <button className="danger" onClick={() => {
                close()
                if (confirm(`Delete "${chat.name}"?`)) onDelete()
              }}>
                <TrashIcon size={15} /> Delete
              </button>
            </>)}
          </div>
        )}
      </div>
    </div>
  )
}

/** Highlight the query inside a snippet. */
function Highlight({ text, query }: { text: string; query: string }) {
  const i = text.toLowerCase().indexOf(query.toLowerCase())
  if (!query || i === -1) return <>{text}</>
  return <>{text.slice(0, i)}<mark>{text.slice(i, i + query.length)}</mark>{text.slice(i + query.length)}</>
}

function useSearch(query: string) {
  const [results, setResults] = useState<SearchResult[] | null>(null)
  useEffect(() => {
    const q = query.trim()
    if (!q) { setResults(null); return }
    let alive = true
    const id = setTimeout(() => {
      api.search(q).then(r => alive && setResults(r)).catch(() => alive && setResults([]))
    }, 180)
    return () => { alive = false; clearTimeout(id) }
  }, [query])
  return results
}

export default function Sidebar({
  open, onToggle, chats, projects, currentChatId, view, onSelect, onNewChat, onDelete, onRename, onPin,
  onGenerateTitle, onMove, onOpenProject, onNewProject, onSchedules, onSettings, displayName, searchSignal,
}: Props) {
  const [search, setSearch] = useState('')
  const [searching, setSearching] = useState(false)
  const [allProjects, setAllProjects] = useState(false)
  const searchInput = useRef<HTMLInputElement>(null)
  const results = useSearch(search)
  const starred = chats.filter(c => c.pinned)
  const recents = chats.filter(c => !c.pinned)
  const initial = (displayName.trim()[0] ?? 'Y').toUpperCase()
  const shownProjects = allProjects ? projects : projects.slice(0, PROJECTS_SHOWN)

  useEffect(() => { if (searchSignal) setSearching(true) }, [searchSignal])
  // Focus explicitly: autoFocus doesn't fire when the box is already open, or when focus is elsewhere.
  useEffect(() => { if (searching) searchInput.current?.focus() }, [searching, searchSignal, open])

  const row = (c: Chat) => (
    <ChatRow
      key={c.id}
      chat={c}
      projects={projects}
      active={c.id === currentChatId && view.kind === 'chat'}
      onSelect={() => onSelect(c.id)}
      onDelete={() => onDelete(c.id)}
      onRename={name => onRename(c.id, name)}
      onPin={() => onPin(c.id)}
      onGenerateTitle={() => onGenerateTitle(c.id)}
      onMove={projectId => onMove(c.id, projectId)}
    />
  )

  const closeSearch = () => { setSearch(''); setSearching(false) }

  if (!open) {
    return (
      <aside className="sidebar collapsed">
        <button className="icon-btn" onClick={onToggle} title="Open sidebar" aria-label="Open sidebar"><PanelIcon /></button>
        <button className="new-chat-mini" onClick={onNewChat} title="New chat" aria-label="New chat"><PlusIcon size={16} /></button>
        <button className="icon-btn" onClick={() => { onToggle(); setSearching(true) }} title="Search" aria-label="Search chats">
          <SearchIcon />
        </button>
        <button className="icon-btn" onClick={onSchedules} title="Scheduled" aria-label="Scheduled tasks"><ClockIcon /></button>
        <div className="sidebar-spacer" />
        <button className="avatar" onClick={onSettings} title="Settings" aria-label="Settings">{initial}</button>
      </aside>
    )
  }

  return (
    <>
      <div className="sidebar-scrim" onClick={onToggle} />
      <aside className="sidebar">
        <div className="sidebar-top">
          <span className="brand">Local LLM</span>
          <button className="icon-btn" onClick={onToggle} title="Close sidebar" aria-label="Close sidebar"><PanelIcon /></button>
        </div>

        <nav className="sidebar-nav">
          <button className="nav-item new-chat" onClick={onNewChat}>
            <span className="new-chat-icon"><PlusIcon size={14} /></span> New chat
          </button>
          {searching ? (
            <div className="sidebar-search">
              <SearchIcon size={16} />
              <input ref={searchInput} placeholder="Search all messages" value={search} onChange={e => setSearch(e.target.value)}
                     onKeyDown={e => { if (e.key === 'Escape') closeSearch() }} />
              <button className="icon-btn small" onClick={closeSearch} aria-label="Close search">
                <CloseIcon size={14} />
              </button>
            </div>
          ) : (
            <button className="nav-item" onClick={() => setSearching(true)}><SearchIcon size={16} /> Search</button>
          )}
          <button className={`nav-item${view.kind === 'schedules' ? ' active' : ''}`} onClick={onSchedules}>
            <ClockIcon size={16} /> Scheduled
          </button>
        </nav>

        <div className="chat-list">
          {results !== null ? (
            <section>
              <h3 className="chat-list-label">{results.length ? `${results.length} result${results.length === 1 ? '' : 's'}` : 'Results'}</h3>
              {results.map(r => (
                <button key={r.id} className="search-hit" onClick={() => { onSelect(r.id, r.index); closeSearch() }}>
                  <span className="search-hit-name"><Highlight text={r.name} query={search.trim()} /></span>
                  {r.snippet && <span className="search-hit-snippet"><Highlight text={r.snippet} query={search.trim()} /></span>}
                </button>
              ))}
              {results.length === 0 && <div className="chat-list-empty">No messages match “{search.trim()}”</div>}
            </section>
          ) : (<>
            <section>
              <div className="chat-list-head">
                <h3 className="chat-list-label">Projects</h3>
                <button className="icon-btn small" onClick={onNewProject} title="New project" aria-label="New project">
                  <PlusIcon size={14} />
                </button>
              </div>
              {shownProjects.map(p => (
                <button key={p.id} className={`project-row${view.kind === 'project' && view.id === p.id ? ' active' : ''}`}
                        onClick={() => onOpenProject(p.id)}>
                  <FolderIcon size={15} /> <span className="chat-row-name">{p.name}</span>
                </button>
              ))}
              {projects.length > PROJECTS_SHOWN && (
                <button className="link-row" onClick={() => setAllProjects(a => !a)}>
                  {allProjects ? 'Show fewer' : `Show all ${projects.length}`}
                </button>
              )}
              {projects.length === 0 && (
                <button className="link-row" onClick={onNewProject}>Group chats with shared instructions and files</button>
              )}
            </section>
            {starred.length > 0 && (
              <section>
                <h3 className="chat-list-label">Starred</h3>
                {starred.map(row)}
              </section>
            )}
            <section>
              <h3 className="chat-list-label">Recents</h3>
              {recents.map(row)}
              {chats.length === 0 && <div className="chat-list-empty">No conversations yet</div>}
            </section>
          </>)}
        </div>

        <div className="sidebar-footer">
          <button className={`account${view.kind === 'settings' ? ' active' : ''}`} onClick={onSettings}>
            <span className="avatar">{initial}</span>
            <span className="account-name">{displayName.trim() || 'You'}</span>
            <SettingsIcon size={16} className="account-gear" />
          </button>
        </div>
      </aside>
    </>
  )
}
