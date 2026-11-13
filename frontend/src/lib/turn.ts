/**
 * Turn state = a pure fold over the task's events. The live view, a reconnect after a
 * dropped connection, and reattaching after a page reload all rebuild state the same way:
 * `events.reduce(applyEvent, initialTurn())`.
 */
import type { StreamEvent } from './sse'
import type { AgentAction, InputRequest, Message, SourceRef, TaskFile, Usage } from '../types'

export interface TurnState {
  taskId?: string
  lastSeq: number
  content: string
  status?: string
  actions: AgentAction[]
  sources: SourceRef[]
  files: TaskFile[]
  pending?: InputRequest
  notice?: { kind: 'error' | 'blocked'; text: string }
  stopped?: boolean
  name?: string
  usage?: Usage
  startedAt: number
  endedAt?: number
  ended: boolean        // task_end seen — nothing more will arrive
}

export const initialTurn = (startedAt = Date.now()): TurnState => ({
  lastSeq: -1, content: '', actions: [], sources: [], files: [], startedAt, ended: false,
})

export function applyEvent(s: TurnState, e: StreamEvent): TurnState {
  if (e.seq !== undefined && e.seq <= s.lastSeq) return s // duplicate after a reconnect
  const next: TurnState = { ...s, lastSeq: e.seq ?? s.lastSeq }
  switch (e.type) {
    case 'task_start':
      return { ...next, taskId: e.task_id }
    case 'status':
      return { ...next, status: e.text }
    case 'action':
      return { ...next, actions: upsert(s.actions, e.action) }
    case 'sources':
      return { ...next, sources: e.sources }
    case 'input_request':
      return { ...next, pending: e.request }
    case 'input_resolved':
      return s.pending?.id === e.request_id ? { ...next, pending: undefined } : next
    case 'token':
      return { ...next, content: s.content + e.content, status: undefined }
    case 'blocked': {
      const label = e.target === 'input' ? 'This message was' : 'This response was'
      return {
        ...next,
        content: e.target === 'output' ? '' : s.content,
        notice: { kind: 'blocked', text: `${label} blocked by safety guardrails${e.reason ? `: ${e.reason}` : '.'}` },
      }
    }
    case 'error':
      return { ...next, notice: { kind: 'error', text: e.message } }
    case 'done':
      return { ...next, name: e.name, files: e.files ?? s.files }
    case 'usage': {
      const { type: _t, seq: _s, ...usage } = e
      return { ...next, usage }
    }
    case 'task_end':
      return {
        ...next,
        ended: true,
        endedAt: Date.now(),
        pending: undefined,
        status: undefined,
        stopped: s.stopped || e.status === 'cancelled',
        notice: s.notice ?? (e.status === 'interrupted'
          ? { kind: 'error', text: 'Interrupted — the server restarted while this was running. Send the message again to retry.' }
          : undefined),
        // Anything still in flight when the task ended didn't finish.
        actions: s.actions.map(a => (a.status === 'running' || a.status === 'waiting'
          ? { ...a, status: e.status === 'cancelled' ? 'denied' : 'failed', ended: a.ended ?? Date.now() }
          : a)),
      }
    default:
      return next
  }
}

function upsert(list: AgentAction[], action: AgentAction): AgentAction[] {
  const i = list.findIndex(a => a.id === action.id)
  if (i === -1) return [...list, action]
  const copy = list.slice()
  copy[i] = { ...list[i], ...action }
  return copy
}

/** Saved messages from older versions stored `steps`; show them as finished actions. */
export function savedActions(m: Message): AgentAction[] {
  if (m.actions?.length) return m.actions
  return (m.steps ?? []).map((st, i) => ({
    id: `legacy-${i}`, tool: st.tool, kind: legacyKind(st.tool), label: st.label, summary: st.summary, status: 'done',
  }))
}

function legacyKind(tool: string): string {
  return ({ news_search: 'news', fetch_page: 'page', get_weather: 'weather' } as Record<string, string>)[tool] ?? 'search'
}

/** The single line shown in the collapsed header while work is in progress. */
export function liveLine(s: Pick<TurnState, 'actions' | 'status' | 'pending'>): string {
  if (s.pending) return s.pending.kind === 'approval' ? 'Waiting for your approval' : 'Waiting for your answer'
  const running = s.actions.filter(a => a.status === 'running')
  if (running.length) {
    const last = running[running.length - 1]
    return running.length > 1 ? `${last.label} (+${running.length - 1} more)` : last.label
  }
  return s.status ?? 'Thinking'
}

export function formatDuration(ms: number): string {
  const s = Math.max(0, Math.round(ms / 1000))
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  return `${m}m ${s % 60}s`
}

export function formatCost(usd: number): string {
  if (usd <= 0) return ''
  return usd < 0.01 ? '<$0.01' : `$${usd.toFixed(2)}`
}
