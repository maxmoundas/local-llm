import type {
  AppSettings, Attachment, Chat, ChatDetail, ChatSettings, HealthStatus, InputRequest, MemoryData, ModelOption,
  PendingFile, Project, ProjectDetail, RestoreResult, Schedule, SearchResult, UsageReport,
} from '../types'

// All paths are relative — Vite proxies /api/* and /health to :8000 in dev.
// In a built+served-from-FastAPI setup they hit the same origin.

/** Fired when the server wants the access token (the app then shows the sign-in screen). */
export const AUTH_EVENT = 'llm-auth-required'

export class ApiError extends Error {
  constructor(message: string, readonly status: number) { super(message) }
}

async function raw(path: string, init?: RequestInit): Promise<Response> {
  const res = await fetch(path, { credentials: 'same-origin', ...init })
  if (res.status === 401 && !path.startsWith('/api/auth')) window.dispatchEvent(new Event(AUTH_EVENT))
  return res
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await raw(path, init)
  if (!res.ok) {
    const body = await res.json().catch(() => null)
    throw new ApiError(body?.detail ?? `Server error ${res.status}`, res.status)
  }
  if (res.status === 204) return undefined as T
  return res.json()
}

const json = (method: string, body?: unknown): RequestInit => ({
  method,
  headers: { 'Content-Type': 'application/json' },
  body: body === undefined ? undefined : JSON.stringify(body),
})

const enc = encodeURIComponent

export const api = {
  health: (): Promise<HealthStatus> => req('/health'),
  authStatus: (): Promise<{ required: boolean; authenticated: boolean }> => req('/api/auth/status'),
  login: (token: string): Promise<{ ok: boolean }> => req('/api/auth/login', json('POST', { token })),
  logout: (): Promise<{ ok: boolean }> => req('/api/auth/logout', json('POST')),

  listModels: (): Promise<ModelOption[]> => req('/api/models'),

  // ── chats ──
  listChats: (): Promise<Chat[]> => req('/api/chats'),
  createChat: (projectId?: string | null): Promise<ChatDetail> =>
    req('/api/chats', json('POST', { project_id: projectId ?? null })),
  getChat: (id: string): Promise<ChatDetail> => req(`/api/chats/${id}`),
  renameChat: (id: string, name: string): Promise<ChatDetail> => req(`/api/chats/${id}`, json('PATCH', { name })),
  moveChat: (id: string, projectId: string | null): Promise<ChatDetail> =>
    req(`/api/chats/${id}`, json('PATCH', { project_id: projectId ?? '' })),
  updateChatSettings: (id: string, settings: ChatSettings): Promise<ChatDetail> =>
    req(`/api/chats/${id}`, json('PATCH', { settings })),
  deleteChat: (id: string): Promise<void> => req(`/api/chats/${id}`, { method: 'DELETE' }),
  pinChat: (id: string): Promise<{ pinned: boolean }> => req(`/api/chats/${id}/pin`, { method: 'POST' }),
  generateTitle: (id: string): Promise<ChatDetail> => req(`/api/chats/${id}/title`, { method: 'POST' }),

  // ── backup ──
  backupUrl: '/api/backup',
  restore: (file: File, mode: 'merge' | 'replace'): Promise<RestoreResult> =>
    req(`/api/restore?mode=${mode}`, { method: 'POST', headers: { 'Content-Type': 'application/zip' }, body: file }),
  switchVersion: (id: string, index: number, version: number): Promise<ChatDetail> =>
    req(`/api/chats/${id}/versions`, json('POST', { index, version })),
  exportUrl: (id: string, format: 'md' | 'json'): string => `/api/chats/${id}/export?format=${format}`,
  search: (q: string): Promise<SearchResult[]> => req(`/api/search?q=${enc(q)}`),

  uploadFile: (chatId: string, f: PendingFile): Promise<Attachment> =>
    req(`/api/chats/${chatId}/files`, json('POST', { name: f.name, mime: f.mime, data_base64: f.base64 })),
  fileUrl: (chatId: string, fileId: string, download = false): string =>
    `/api/chats/${chatId}/files/${fileId}${download ? '?download=true' : ''}`,

  /** These return the raw Response so the caller can read the SSE body stream. */
  sendMessage: (id: string, content: string, attachments: string[], modelId?: string): Promise<Response> =>
    raw(`/api/chats/${id}/message`, json('POST', { content, attachments, model_id: modelId ?? null })),
  regenerate: (id: string, index: number | null, modelId?: string): Promise<Response> =>
    raw(`/api/chats/${id}/regenerate`, json('POST', { index, model_id: modelId ?? null })),
  editMessage: (id: string, index: number, content: string, modelId?: string): Promise<Response> =>
    raw(`/api/chats/${id}/edit`, json('POST', { index, content, model_id: modelId ?? null })),

  /** The task still running for a chat (to reattach after a reload), or null. */
  activeTask: (chatId: string): Promise<{ task_id: string; pending: InputRequest | null } | null> =>
    raw(`/api/chats/${chatId}/active`).then(r => (r.ok ? r.json() : null)),

  /** Resume a task's event stream after `after` (the last seq seen). */
  taskEvents: (taskId: string, after: number, signal?: AbortSignal): Promise<Response> =>
    raw(`/api/tasks/${taskId}/events?after=${after}`, { signal }),

  cancelTask: (taskId: string): Promise<{ cancelled: boolean }> =>
    req(`/api/tasks/${taskId}/cancel`, { method: 'POST' }),

  replyTask: (taskId: string, reply: { request_id: string; answer?: string; approved?: boolean }): Promise<boolean> =>
    raw(`/api/tasks/${taskId}/reply`, json('POST', reply)).then(r => r.ok),

  taskFileUrl: (taskId: string, path: string): string =>
    `/api/tasks/${taskId}/files/${path.split('/').map(enc).join('/')}`,

  // ── settings & memory ──
  getSettings: (): Promise<AppSettings> => req('/api/settings'),
  updateSettings: (data: Partial<AppSettings>): Promise<AppSettings> => req('/api/settings', json('PATCH', data)),
  getMemory: (): Promise<MemoryData> => req('/api/memory'),
  updateMemory: (content: string): Promise<MemoryData> => req('/api/memory', json('PATCH', { content })),

  // ── projects ──
  listProjects: (): Promise<Project[]> => req('/api/projects'),
  getProject: (id: string): Promise<ProjectDetail> => req(`/api/projects/${id}`),
  createProject: (data: { name: string; description?: string; instructions?: string }): Promise<Project> =>
    req('/api/projects', json('POST', data)),
  updateProject: (id: string, data: Partial<Pick<Project, 'name' | 'description' | 'instructions'>>): Promise<Project> =>
    req(`/api/projects/${id}`, json('PATCH', data)),
  deleteProject: (id: string): Promise<void> => req(`/api/projects/${id}`, { method: 'DELETE' }),
  addProjectFile: (id: string, f: PendingFile): Promise<Attachment> =>
    req(`/api/projects/${id}/files`, json('POST', { name: f.name, mime: f.mime, data_base64: f.base64 })),
  removeProjectFile: (id: string, fileId: string): Promise<void> =>
    req(`/api/projects/${id}/files/${fileId}`, { method: 'DELETE' }),
  projectFileUrl: (id: string, fileId: string): string => `/api/projects/${id}/files/${fileId}`,
  updateProjectMemory: (id: string, content: string): Promise<MemoryData> =>
    req(`/api/projects/${id}/memory`, json('PATCH', { content })),

  // ── schedules ──
  listSchedules: (): Promise<Schedule[]> => req('/api/schedules'),
  createSchedule: (data: Partial<Schedule>): Promise<Schedule> => req('/api/schedules', json('POST', data)),
  updateSchedule: (id: string, data: Partial<Schedule>): Promise<Schedule> =>
    req(`/api/schedules/${id}`, json('PATCH', data)),
  deleteSchedule: (id: string): Promise<void> => req(`/api/schedules/${id}`, { method: 'DELETE' }),
  runSchedule: (id: string): Promise<{ chat_id: string; task_id: string }> =>
    req(`/api/schedules/${id}/run`, { method: 'POST' }),

  // ── usage & voice ──
  usage: (days = 30): Promise<UsageReport> => req(`/api/usage?days=${days}`),
  transcribe: (audioBase64: string, mime: string): Promise<{ text: string }> =>
    req('/api/transcribe', json('POST', { audio_base64: audioBase64, mime })),
}

/** Read a picked file into a PendingFile (base64 for upload, data URL for image previews). */
export function readFile(file: File): Promise<PendingFile> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      const dataUrl = reader.result as string
      const isImage = /^image\/(png|jpeg|gif|webp)$/.test(file.type)
      resolve({
        key: `${file.name}-${file.size}-${Math.random().toString(36).slice(2, 8)}`,
        name: file.name || (isImage ? 'pasted-image.png' : 'file'),
        size: file.size,
        mime: file.type || 'application/octet-stream',
        base64: dataUrl.split(',')[1] ?? '',
        kind: isImage ? 'image' : 'document',
        dataUrl: isImage ? dataUrl : undefined,
      })
    }
    reader.onerror = reject
    reader.readAsDataURL(file)
  })
}

export function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}
