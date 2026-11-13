import type { AgentAction, InputRequest, MemoryChange, SourceRef, TaskFile, Usage } from '../types'

/** Events of one chat turn (a background task). Every event carries a `seq` for resuming. */
export type StreamEvent = { seq?: number } & (
  | { type: 'task_start'; task_id: string; chat_id?: string }
  | { type: 'status'; text: string }
  | { type: 'action'; action: AgentAction }
  | { type: 'sources'; sources: SourceRef[] }
  | { type: 'input_request'; request: InputRequest }
  | { type: 'input_resolved'; request_id: string }
  | { type: 'token'; content: string }
  | { type: 'blocked'; target: 'input' | 'output'; reason?: string }
  | { type: 'error'; message: string }
  | ({ type: 'memory' } & MemoryChange)
  | { type: 'done'; name: string; files?: TaskFile[] }
  | ({ type: 'usage' } & Usage)
  | { type: 'task_end'; status: 'done' | 'failed' | 'cancelled' | 'interrupted' }
)

/** Read a text/event-stream response body, yielding each `data:` JSON event (comments/pings skipped). */
export async function* readEvents(res: Response): AsyncGenerator<StreamEvent> {
  const reader = res.body!.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const parts = buffer.split('\n\n')
    buffer = parts.pop()!
    for (const part of parts) {
      for (const line of part.split('\n')) {
        if (!line.startsWith('data: ')) continue
        try {
          yield JSON.parse(line.slice(6))
        } catch { /* malformed line — skip */ }
      }
    }
  }
}
