import { useEffect, useRef, useState } from 'react'
import type { Chat } from '../types'
import {
  CloseIcon, DotsIcon, PanelIcon, PencilIcon, PlusIcon, SearchIcon, SettingsIcon, StarIcon, TrashIcon,
} from './Icons'

interface Props {
  open: boolean
  onToggle: () => void
  chats: Chat[]
  currentChatId: string | null
  onSelect: (id: string) => void
  onNewChat: () => void
  onDelete: (id: string) => void
  onRename: (id: string, name: string) => void
  onPin: (id: string) => void
  onSettings: () => void
  settingsActive: boolean
  displayName: string
}

function ChatRow({ chat, active, onSelect, onDelete, onRename, onPin }: {
  chat: Chat
  active: boolean
  onSelect: () => void
  onDelete: () => void
  onRename: (name: string) => void
  onPin: () => void
}) {
  const [menuOpen, setMenuOpen] = useState(false)
  const [editing, setEditing] = useState(false)
  const [name, setName] = useState(chat.name)
  const menuRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!menuOpen) return
    const onDown = (e: MouseEvent) => { if (!menuRef.current?.contains(e.target as Node)) setMenuOpen(false) }
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

  return (
    <div className={`chat-row${active ? ' active' : ''}${menuOpen ? ' menu-open' : ''}`} onClick={onSelect}>
      <span className="chat-row-name" onDoubleClick={() => setEditing(true)}>{chat.name}</span>
      <div className="chat-row-menu" ref={menuRef} onClick={e => e.stopPropagation()}>
        <button className="icon-btn small" onClick={() => setMenuOpen(o => !o)} aria-label="Chat options">
          <DotsIcon size={16} />
        </button>
        {menuOpen && (
          <div className="popover row-menu">
            <button onClick={() => { setMenuOpen(false); onPin() }}>
              <StarIcon size={15} filled={chat.pinned} /> {chat.pinned ? 'Unstar' : 'Star'}
            </button>
            <button onClick={() => { setMenuOpen(false); setEditing(true) }}>
              <PencilIcon size={15} /> Rename
            </button>
            <button className="danger" onClick={() => {
              setMenuOpen(false)
              if (confirm(`Delete "${chat.name}"?`)) onDelete()
            }}>
              <TrashIcon size={15} /> Delete
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

export default function Sidebar({
  open, onToggle, chats, currentChatId, onSelect, onNewChat, onDelete, onRename, onPin,
  onSettings, settingsActive, displayName,
}: Props) {
  const [search, setSearch] = useState('')
  const [searching, setSearching] = useState(false)
  const query = search.trim().toLowerCase()
  const filtered = query
    ? chats.filter(c => c.name.toLowerCase().includes(query) || c.preview.toLowerCase().includes(query))
    : chats
  const starred = filtered.filter(c => c.pinned)
  const recents = filtered.filter(c => !c.pinned)
  const initial = (displayName.trim()[0] ?? 'Y').toUpperCase()

  const row = (c: Chat) => (
    <ChatRow
      key={c.id}
      chat={c}
      active={c.id === currentChatId && !settingsActive}
      onSelect={() => onSelect(c.id)}
      onDelete={() => onDelete(c.id)}
      onRename={name => onRename(c.id, name)}
      onPin={() => onPin(c.id)}
    />
  )

  if (!open) {
    return (
      <aside className="sidebar collapsed">
        <button className="icon-btn" onClick={onToggle} title="Open sidebar" aria-label="Open sidebar"><PanelIcon /></button>
        <button className="new-chat-mini" onClick={onNewChat} title="New chat" aria-label="New chat"><PlusIcon size={16} /></button>
        <button className="icon-btn" onClick={() => { onToggle(); setSearching(true) }} title="Search" aria-label="Search chats">
          <SearchIcon />
        </button>
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
              <input autoFocus placeholder="Search chats" value={search} onChange={e => setSearch(e.target.value)}
                     onKeyDown={e => { if (e.key === 'Escape') { setSearch(''); setSearching(false) } }} />
              <button className="icon-btn small" onClick={() => { setSearch(''); setSearching(false) }} aria-label="Close search">
                <CloseIcon size={14} />
              </button>
            </div>
          ) : (
            <button className="nav-item" onClick={() => setSearching(true)}><SearchIcon size={16} /> Search</button>
          )}
        </nav>

        <div className="chat-list">
          {starred.length > 0 && (
            <section>
              <h3 className="chat-list-label">Starred</h3>
              {starred.map(row)}
            </section>
          )}
          <section>
            <h3 className="chat-list-label">Recents</h3>
            {recents.map(row)}
            {filtered.length === 0 && (
              <div className="chat-list-empty">{query ? 'No chats match your search' : 'No conversations yet'}</div>
            )}
          </section>
        </div>

        <div className="sidebar-footer">
          <button className={`account${settingsActive ? ' active' : ''}`} onClick={onSettings}>
            <span className="avatar">{initial}</span>
            <span className="account-name">{displayName.trim() || 'You'}</span>
            <SettingsIcon size={16} className="account-gear" />
          </button>
        </div>
      </aside>
    </>
  )
}
