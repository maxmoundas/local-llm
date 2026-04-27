import type { Chat } from '../types'

interface Props {
  chats: Chat[]
  currentChatId: string | null
  onSelect: (id: string) => void
  onCreate: () => void
  onDelete: (id: string) => void
}

export default function Sidebar({ chats, currentChatId, onSelect, onCreate, onDelete }: Props) {
  const handleDelete = (e: React.MouseEvent, id: string, name: string) => {
    e.stopPropagation()
    if (confirm(`Delete "${name}"?`)) onDelete(id)
  }

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="brand">
          <div className="brand-dot" />
          Local LLM
        </div>
        <button className="new-chat-btn" onClick={onCreate} title="New chat">+</button>
      </div>

      <div className="chat-list">
        {chats.map(chat => (
          <div
            key={chat.id}
            className={`chat-item${chat.id === currentChatId ? ' active' : ''}`}
            onClick={() => onSelect(chat.id)}
          >
            <div className="chat-item-text">
              <div className="chat-item-name">{chat.name}</div>
              {chat.preview && (
                <div className="chat-item-preview">{chat.preview}</div>
              )}
            </div>
            <button
              className="chat-delete-btn"
              onClick={e => handleDelete(e, chat.id, chat.name)}
              title="Delete chat"
            >
              ✕
            </button>
          </div>
        ))}
      </div>
    </aside>
  )
}
