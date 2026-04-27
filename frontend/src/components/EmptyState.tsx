interface Props { onCreate: () => void }

export default function EmptyState({ onCreate }: Props) {
  return (
    <div className="empty-state">
      <div className="empty-icon">◈</div>
      <h2>Local LLM Chat</h2>
      <p>Select a conversation or start a new one</p>
      <button className="empty-new-btn" onClick={onCreate}>New Chat</button>
    </div>
  )
}
