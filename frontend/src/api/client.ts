import type { AppSettings, Chat, ChatDetail, HealthStatus, MemoryData, ModelOption } from '../types'

// All paths are relative — Vite proxies /api/* and /health to :8000 in dev.
// In a built+served-from-FastAPI setup they hit the same origin.

export const api = {
  health: (): Promise<HealthStatus> =>
    fetch('/health').then(r => r.json()),

  listModels: (): Promise<ModelOption[]> =>
    fetch('/api/models').then(r => r.json()),

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

  pinChat: (id: string): Promise<{ pinned: boolean }> =>
    fetch(`/api/chats/${id}/pin`, { method: 'POST' }).then(r => r.json()),

  /** Returns the raw Response so the caller can read the SSE body stream. */
  sendMessage: (id: string, content: string, imageBase64?: string, modelId?: string, signal?: AbortSignal): Promise<Response> =>
    fetch(`/api/chats/${id}/message`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content, image_base64: imageBase64 ?? null, model_id: modelId ?? null }),
      signal,
    }),

  getSettings: (): Promise<AppSettings> =>
    fetch('/api/settings').then(r => r.json()),

  updateSettings: (data: Partial<Omit<AppSettings, 'models' | 'jev_configured'>>): Promise<AppSettings> =>
    fetch('/api/settings', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    }).then(r => r.json()),

  getMemory: (): Promise<MemoryData> =>
    fetch('/api/memory').then(r => r.json()),

  updateMemory: (content: string): Promise<MemoryData> =>
    fetch('/api/memory', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content }),
    }).then(r => r.json()),
}
