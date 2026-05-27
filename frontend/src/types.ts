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
  is_cheapest?: boolean
}

export interface AppSettings {
  cheapest_model_id: string | null
  system_prompt: string
  default_system_prompt: string
  memory_enabled: boolean
  guardrails_enabled: boolean
  guardrails_input: boolean
  guardrails_output: boolean
  guardrails_prompt: string
  guardrails_topics: string[]
  default_guardrail_prompt: string
  models: ModelOption[]
}

export interface MemoryData {
  content: string
}

export interface AttachedImage {
  base64: string    // sent to API
  dataUrl: string   // shown in UI preview and message bubble
  name: string
  size: number
}
