export interface Chat {
  id: string
  name: string
  created_at: string
  updated_at: string
  preview: string
}

export interface Message {
  role: 'system' | 'user' | 'assistant'
  content: string
}

export interface ChatDetail extends Chat {
  messages: Message[]
}

export interface HealthStatus {
  status: 'ok' | 'loading'
  model: string | null
}
