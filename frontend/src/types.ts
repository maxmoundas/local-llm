export interface Chat {
  id: string
  name: string
  created_at: string
  updated_at: string
  preview: string
  model_id?: string
  pinned: boolean
}

export interface ResearchStep {
  tool: string
  label: string
  summary?: string
}

export interface SourceRef {
  title: string
  url: string
}

export interface Message {
  role: 'system' | 'user' | 'assistant'
  content: string
  steps?: ResearchStep[]
  sources?: SourceRef[]
  stopped?: boolean
}

export interface ChatDetail extends Chat {
  messages: Message[]
  model_id?: string
}

export interface HealthStatus {
  status: 'ok' | 'loading'
  model: string | null
  local_model_loaded: boolean
  jev_available: boolean
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
  guardrails_policy: 'strict' | 'permissive'
  default_guardrail_prompt: string
  agent_enabled: boolean
  agent_max_steps: number
  jev_enabled: boolean
  jev_model: string
  jev_routing: boolean
  jev_guardrails: boolean
  jev_memory: boolean
  jev_configured: boolean
  display_name: string
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
