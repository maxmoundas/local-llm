import type { ResearchStep, SourceRef } from '../types'

export type StreamEvent =
  | { type: 'status'; text: string }
  | { type: 'step'; step: ResearchStep }
  | { type: 'sources'; sources: SourceRef[] }
  | { type: 'token'; content: string }
  | { type: 'done'; name: string }
  | { type: 'error'; message: string }
  | { type: 'blocked'; target: 'input' | 'output'; reason?: string }

/** Read a text/event-stream response body, yielding each `data:` JSON event. */
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
