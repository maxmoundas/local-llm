import { describe, expect, it } from 'vitest'
import type { StreamEvent } from './sse'
import { applyEvent, formatCost, formatDuration, initialTurn, liveLine, savedActions } from './turn'
import type { AgentAction } from '../types'

const act = (over: Partial<AgentAction> = {}): AgentAction => ({
  id: 'a1', tool: 'web_search', kind: 'search', label: 'Searching for “x”', status: 'running', ...over,
})

const fold = (events: StreamEvent[]) => events.reduce(applyEvent, initialTurn(0))

describe('applyEvent', () => {
  it('records a memory change', () => {
    const s = fold([{ seq: 0, type: 'memory', added: ['Lives in Boston'], removed: [] }])
    expect(s.memory).toEqual({ added: ['Lives in Boston'], removed: [] })
  })

  it('builds a full turn from its events', () => {
    const s = fold([
      { seq: 0, type: 'task_start', task_id: 't1' },
      { seq: 1, type: 'status', text: 'Thinking' },
      { seq: 2, type: 'action', action: act() },
      { seq: 3, type: 'action', action: act({ status: 'done', label: 'Searched for “x”', summary: '6 results' }) },
      { seq: 4, type: 'sources', sources: [{ title: 'A', url: 'https://a.com' }] },
      { seq: 5, type: 'token', content: 'Hello ' },
      { seq: 6, type: 'token', content: 'world' },
      { seq: 7, type: 'done', name: 'Chat', files: [{ path: 'out.csv', size: 10 }] },
      { seq: 8, type: 'usage', input_tokens: 10, output_tokens: 5, cost_usd: 0.001, calls: 2 },
      { seq: 9, type: 'task_end', status: 'done' },
    ])
    expect(s.taskId).toBe('t1')
    expect(s.actions).toHaveLength(1)
    expect(s.actions[0]).toMatchObject({ status: 'done', summary: '6 results' })
    expect(s.content).toBe('Hello world')
    expect(s.status).toBeUndefined()
    expect(s.files[0].path).toBe('out.csv')
    expect(s.usage?.calls).toBe(2)
    expect(s.ended).toBe(true)
    expect(s.lastSeq).toBe(9)
  })

  it('ignores events replayed after a reconnect', () => {
    const once = fold([{ seq: 0, type: 'token', content: 'a' }, { seq: 1, type: 'token', content: 'b' }])
    const again = applyEvent(applyEvent(once, { seq: 1, type: 'token', content: 'b' }), { seq: 2, type: 'token', content: 'c' })
    expect(again.content).toBe('abc')
  })

  it('tracks a pending question until it is resolved', () => {
    const asked = fold([{ seq: 0, type: 'input_request', request: { id: 'r1', kind: 'question', question: 'Which?' } }])
    expect(asked.pending?.id).toBe('r1')
    expect(applyEvent(asked, { seq: 1, type: 'input_resolved', request_id: 'other' }).pending).toBeDefined()
    expect(applyEvent(asked, { seq: 1, type: 'input_resolved', request_id: 'r1' }).pending).toBeUndefined()
  })

  it('marks unfinished actions when the task is cancelled', () => {
    const s = fold([
      { seq: 0, type: 'action', action: act() },
      { seq: 1, type: 'task_end', status: 'cancelled' },
    ])
    expect(s.stopped).toBe(true)
    expect(s.actions[0].status).toBe('denied')
  })

  it('explains an interrupted task', () => {
    expect(fold([{ seq: 0, type: 'task_end', status: 'interrupted' }]).notice?.kind).toBe('error')
  })

  it('clears blocked output', () => {
    const s = fold([{ seq: 0, type: 'token', content: 'bad' }, { seq: 1, type: 'blocked', target: 'output' }])
    expect(s.content).toBe('')
    expect(s.notice?.text).toMatch(/blocked/)
  })
})

describe('liveLine', () => {
  it('prefers a waiting request, then running actions, then the status', () => {
    expect(liveLine({ actions: [], status: 'Thinking' })).toBe('Thinking')
    expect(liveLine({ actions: [act()], status: 'Thinking' })).toBe('Searching for “x”')
    expect(liveLine({ actions: [act(), act({ id: 'a2', label: 'Reading a.com' })] })).toBe('Reading a.com (+1 more)')
    expect(liveLine({ actions: [act()], pending: { id: 'r', kind: 'approval' } })).toBe('Waiting for your approval')
  })
})

describe('helpers', () => {
  it('converts legacy research steps', () => {
    const a = savedActions({ role: 'assistant', content: '', steps: [{ tool: 'fetch_page', label: 'Read a.com' }] })
    expect(a[0]).toMatchObject({ kind: 'page', status: 'done', label: 'Read a.com' })
  })

  it('formats durations and costs', () => {
    expect(formatDuration(4_200)).toBe('4s')
    expect(formatDuration(72_000)).toBe('1m 12s')
    expect(formatCost(0)).toBe('')
    expect(formatCost(0.004)).toBe('<$0.01')
    expect(formatCost(0.123)).toBe('$0.12')
  })
})
