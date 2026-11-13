export interface Chat {
  id: string
  name: string
  created_at: string
  updated_at: string
  preview: string
  model_id?: string
  pinned: boolean
}

/** Saved by older versions; converted to AgentAction on load. */
export interface ResearchStep {
  tool: string
  label: string
  summary?: string
}

export type ActionStatus = 'waiting' | 'running' | 'done' | 'failed' | 'denied'

/** One tool call the agent made — the unit of the activity timeline. */
export interface AgentAction {
  id: string
  tool: string
  kind: string          // icon family: search | news | page | weather | code | file | connector | question
  label: string         // one-line, user-facing: present tense while running, past/plain when done
  status: ActionStatus
  summary?: string      // short outcome, e.g. "6 results", "exit 1"
  started?: number      // epoch ms
  ended?: number | null
}

export interface TaskFile {
  path: string
  size: number
}

export interface InputRequest {
  id: string
  kind: 'question' | 'approval'
  question?: string
  tool?: string
  label?: string
  arguments?: Record<string, unknown>
  action_id?: string
}

export interface Usage {
  input_tokens: number
  output_tokens: number
  cached_input_tokens?: number
  cost_usd: number
  calls: number
}

export interface SourceRef {
  title: string
  url: string
}

export interface Message {
  role: 'system' | 'user' | 'assistant'
  content: string
  steps?: ResearchStep[]      // legacy
  actions?: AgentAction[]
  sources?: SourceRef[]
  files?: TaskFile[]
  task_id?: string
  elapsed_ms?: number
  usage?: Usage
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
  tools?: 'native' | 'prompted' | 'auto'
  vision?: boolean
  context_window?: number
  input_price?: number
  output_price?: number
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
  agent_model_id: string | null
  task_budget_usd: number | null
  task_max_minutes: number | null
  code_execution: 'auto' | 'docker' | 'local' | 'off'
  sandbox_network: boolean
  sandbox_active: string
  connectors: Record<string, string>
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
