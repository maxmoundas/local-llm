export interface Chat {
  id: string
  name: string
  created_at: string
  updated_at: string
  preview: string
  model_id?: string
  pinned: boolean
  project_id?: string | null
  schedule_id?: string | null
}

/** A file attached to a message or a project (stored by the backend). */
export interface Attachment {
  id: string
  name: string
  mime: string
  size: number
  kind: 'image' | 'document'
  chars?: number
}

/** A file picked in the composer, not uploaded yet. */
export interface PendingFile {
  key: string
  name: string
  size: number
  mime: string
  base64: string
  kind: 'image' | 'document'
  dataUrl?: string   // images: local preview
}

export interface ChatSettings {
  system_prompt?: string
  temperature?: number
  effort?: 'low' | 'medium' | 'high'
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
  reason?: string   // why approval is needed when the tool itself doesn't require it
}

/** Facts the memory system added or removed because of one message. */
export interface MemoryChange {
  added: string[]
  removed: string[]
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
  memory?: MemoryChange
  model_id?: string
  attachments?: Attachment[]
  versions?: Message[][]     // the other versions of the conversation from this message on
  version_index?: number     // where this one sits among all versions
}

export interface ChatDetail extends Chat {
  messages: Message[]
  model_id?: string
  settings?: ChatSettings
}

export interface SearchResult extends Chat {
  index: number | null
  role: 'user' | 'assistant' | null
  snippet: string
}

export interface Project {
  id: string
  name: string
  description: string
  instructions: string
  files: Attachment[]
  created_at: string
  updated_at: string
  chat_count: number
}

export interface ProjectDetail extends Project {
  chats: Chat[]
  memory: string
}

export type ScheduleKind = 'daily' | 'weekly' | 'interval'

export interface Schedule {
  id: string
  name: string
  prompt: string
  model_id: string | null
  project_id: string | null
  kind: ScheduleKind
  time?: string | null
  days?: number[] | null
  every_minutes?: number | null
  enabled: boolean
  chat_id: string | null
  created_at: string
  last_run_at: string | null
  next_run_at: string | null
  last_task_id: string | null
  description: string
}

export interface UsageBucket {
  cost_usd: number
  input_tokens: number
  output_tokens: number
  turns: number
}

export interface UsageReport {
  days: (UsageBucket & { date: string })[]
  models: { model_id: string; name: string; cost_usd: number; input_tokens: number; output_tokens: number; calls: number }[]
  chats: { chat_id: string | null; name: string; cost_usd: number; turns: number }[]
  total: UsageBucket
  month_to_date_usd: number
  monthly_budget_usd: number | null
}

export interface HealthStatus {
  status: 'ok' | 'loading'
  model: string | null
  local_model_enabled?: boolean  // false when the built-in model is turned off (never loads)
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
  max_output_tokens?: number
  input_price?: number
  output_price?: number
  temperature?: boolean   // accepts a sampling temperature
  effort?: boolean        // accepts a reasoning-effort level
  error?: string          // why the model can't be used right now (e.g. a rejected API key)
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
  agent_guard_untrusted: boolean   // ask before code/connectors send data out after reading web content
  workspace_retention_days: number  // delete agent workspaces older than this (0 = keep)
  sandbox_active: string
  connectors: Record<string, string>
  jev_enabled: boolean
  jev_model: string
  jev_routing: boolean
  jev_guardrails: boolean
  jev_memory: boolean
  jev_configured: boolean
  display_name: string
  monthly_budget_usd: number | null
  whisper_model: 'tiny' | 'base' | 'small' | 'medium' | 'large-v3'
  transcription_available: boolean
  access_token_set: boolean
  models: ModelOption[]
}

export interface RestoreResult {
  mode: 'merge' | 'replace'
  chats_restored: number
  chats_skipped: number
  projects_restored: number
  projects_skipped: number
  memory_restored?: boolean
  settings_restored?: boolean
  schedules_restored?: number
}

export interface MemoryData {
  content: string
}

/** Something shown in the side preview panel. */
export type Preview =
  | { kind: 'html' | 'svg' | 'mermaid' | 'text' | 'markdown' | 'csv'; title: string; content: string }
  | { kind: 'image'; title: string; url: string }
  | { kind: 'file'; title: string; url: string }   // fetched, then shown by type
