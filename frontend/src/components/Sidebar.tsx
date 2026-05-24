import { useState, useRef } from 'react'
import type { Chat } from '../types'

interface Props {
  chats: Chat[]
  currentChatId: string | null
  onSelect: (id: string) => void
  onCreate: () => void
  onDelete: (id: string) => void
  onRename: (id: string, name: string) => void
  onPin: (id: string) => void
}

function modelLabel(modelId: string | undefined): string | null {
  if (!modelId) return null
  if (modelId.startsWith('google/') || modelId.includes('gemma')) return 'Local'
  if (modelId.includes('haiku'))  return 'Haiku'
  if (modelId.includes('sonnet')) return 'Sonnet'
  if (modelId.includes('opus'))   return 'Opus'
  if (modelId.includes('mini'))   return 'GPT mini'
  if (modelId.startsWith('gpt-')) return 'GPT'
  return null
}

function modelProvider(modelId: string | undefined): string {
  if (!modelId) return ''
  if (modelId.startsWith('google/') || modelId.includes('gemma')) return 'local'
  if (modelId.includes('claude')) return 'anthropic'
  if (modelId.startsWith('gpt-')) return 'openai'
  return ''
}

function groupByDate(chats: Chat[]): { label: string; chats: Chat[] }[] {
  const now = new Date()
  const today     = new Date(now.getFullYear(), now.getMonth(), now.getDate())
  const yesterday = new Date(today.getTime() - 86_400_000)
  const week      = new Date(today.getTime() - 7 * 86_400_000)

  const buckets: { label: string; test: (d: Date) => boolean }[] = [
    { label: 'Today',            test: d => d >= today },
    { label: 'Yesterday',        test: d => d >= yesterday && d < today },
    { label: 'Previous 7 Days',  test: d => d >= week && d < yesterday },
    { label: 'Older',            test: d => d < week },
  ]

  return buckets
    .map(b => ({ label: b.label, chats: chats.filter(c => b.test(new Date(c.updated_at))) }))
    .filter(g => g.chats.length > 0)
}

export default function Sidebar({ chats, currentChatId, onSelect, onCreate, onDelete, onRename, onPin }: Props) {
  const [search, setSearch]     = useState('')
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editName, setEditName]   = useState('')
  const editRef = useRef<HTMLInputElement>(null)

  const query    = search.trim().toLowerCase()
  const filtered = query
    ? chats.filter(c =>
        c.name.toLowerCase().includes(query) ||
        c.preview.toLowerCase().includes(query)
      )
    : chats

  const pinned   = filtered.filter(c => c.pinned)
  const unpinned = filtered.filter(c => !c.pinned)
  const groups   = groupByDate(unpinned)

  const startEdit = (e: React.MouseEvent, chat: Chat) => {
    e.stopPropagation()
    setEditingId(chat.id)
    setEditName(chat.name)
    setTimeout(() => { editRef.current?.select() }, 0)
  }

  const commitEdit = (id: string) => {
    const trimmed = editName.trim()
    const original = chats.find(c => c.id === id)?.name ?? ''
    if (trimmed && trimmed !== original) onRename(id, trimmed)
    setEditingId(null)
  }

  const handleDelete = (e: React.MouseEvent, id: string, name: string) => {
    e.stopPropagation()
    if (confirm(`Delete "${name}"?`)) onDelete(id)
  }

  const renderChat = (chat: Chat) => {
    const label    = modelLabel(chat.model_id)
    const provider = modelProvider(chat.model_id)
    const isActive  = chat.id === currentChatId
    const isEditing = editingId === chat.id

    return (
      <div
        key={chat.id}
        className={`chat-item${isActive ? ' active' : ''}`}
        onClick={() => { if (!isEditing) onSelect(chat.id) }}
      >
        <div className="chat-item-text">
          {isEditing ? (
            <input
              ref={editRef}
              className="chat-item-name-input"
              value={editName}
              autoFocus
              onChange={e => setEditName(e.target.value)}
              onBlur={() => commitEdit(chat.id)}
              onKeyDown={e => {
                if (e.key === 'Enter')  e.currentTarget.blur()
                if (e.key === 'Escape') setEditingId(null)
                e.stopPropagation()
              }}
              onClick={e => e.stopPropagation()}
            />
          ) : (
            <div className="chat-item-name" onDoubleClick={e => startEdit(e, chat)}>
              {chat.name}
            </div>
          )}
          <div className="chat-item-meta">
            {chat.preview && <span className="chat-item-preview">{chat.preview}</span>}
            {label && (
              <span className={`chat-model-badge provider-${provider}`}>{label}</span>
            )}
          </div>
        </div>

        <div className="chat-item-actions">
          <button
            className={`chat-pin-btn${chat.pinned ? ' pinned' : ''}`}
            onClick={e => { e.stopPropagation(); onPin(chat.id) }}
            title={chat.pinned ? 'Unpin' : 'Pin'}
          >
            {chat.pinned ? '★' : '☆'}
          </button>
          <button
            className="chat-delete-btn"
            onClick={e => handleDelete(e, chat.id, chat.name)}
            title="Delete chat"
          >
            ✕
          </button>
        </div>
      </div>
    )
  }

  const renderGroup = (label: string, items: Chat[]) => (
    <div key={label} className="chat-group">
      <div className="chat-group-label">{label}</div>
      {items.map(renderChat)}
    </div>
  )

  const empty = pinned.length === 0 && unpinned.length === 0

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="brand">
          <div className="brand-dot" />
          Local LLM
        </div>
        <button className="new-chat-btn" onClick={onCreate} title="New chat">+</button>
      </div>

      <div className="sidebar-search">
        <input
          className="search-input"
          placeholder="Search chats…"
          value={search}
          onChange={e => setSearch(e.target.value)}
        />
        {search && (
          <button className="search-clear" onClick={() => setSearch('')}>✕</button>
        )}
      </div>

      <div className="chat-list">
        {empty && (
          <div className="sidebar-no-results">
            {query ? 'No chats match your search' : 'No conversations yet'}
          </div>
        )}
        {pinned.length > 0 && renderGroup('Pinned', pinned)}
        {groups.map(g => renderGroup(g.label, g.chats))}
      </div>
    </aside>
  )
}
