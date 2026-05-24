export interface Chat {
  id: string
  name: string
  created_at: string
  updated_at: string
  preview: string
  model_id?: string
  pinned: boolean
}

export interface Message {
  role: 'system' | 'user' | 'assistant'
  content: string
}

export interface ChatDetail extends Chat {
  messages: Message[]
  model_id?: string
}

export interface HealthStatus {
  status: 'ok' | 'loading'
  model: string | null
  local_model_loaded: boolean
}

export interface ModelOption {
  id: string
  name: string
  provider: string
}

export interface AttachedImage {
  base64: string    // sent to API
  dataUrl: string   // shown in UI preview and message bubble
  name: string
  size: number
}
