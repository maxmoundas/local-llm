import type { Chat, ChatDetail, HealthStatus } from '../types'

// All paths are relative — Vite proxies /api/* and /health to :8000 in dev.
// In a built+served-from-FastAPI setup they hit the same origin.

export const api = {
  health: (): Promise<HealthStatus> =>
    fetch('/health').then(r => r.json()),

  listChats: (): Promise<Chat[]> =>
    fetch('/api/chats').then(r => r.json()),

  createChat: (): Promise<ChatDetail> =>
    fetch('/api/chats', { method: 'POST' }).then(r => r.json()),

  getChat: (id: string): Promise<ChatDetail> =>
    fetch(`/api/chats/${id}`).then(r => r.json()),

  renameChat: (id: string, name: string): Promise<Chat> =>
    fetch(`/api/chats/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    }).then(r => r.json()),

  deleteChat: (id: string): Promise<void> =>
    fetch(`/api/chats/${id}`, { method: 'DELETE' }).then(() => undefined),

  /** Returns the raw Response so the caller can read the SSE body stream. */
  sendMessage: (id: string, content: string): Promise<Response> =>
    fetch(`/api/chats/${id}/message`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content }),
    }),
}
