import { describe, expect, it } from 'vitest'
import { parseCsv } from './preview'
import { normalizeMath } from '../components/Markdown'
import { fromSaved } from '../components/ChatView'

describe('parseCsv', () => {
  it('handles quotes, escaped quotes, and CRLF', () => {
    expect(parseCsv('a,b\r\n"x, y","he said ""hi"""\n1,2')).toEqual([['a', 'b'], ['x, y', 'he said "hi"'], ['1', '2']])
  })
  it('supports tabs', () => {
    expect(parseCsv('a\tb\n1\t2', '\t')).toEqual([['a', 'b'], ['1', '2']])
  })
})

describe('normalizeMath', () => {
  it('converts \\( \\) and \\[ \\] delimiters', () => {
    expect(normalizeMath('area \\(\\pi r^2\\)')).toBe('area $\\pi r^2$')
    expect(normalizeMath('\\[E = mc^2\\]')).toContain('$$\nE = mc^2\n$$')
  })
  it('makes a one-line $$ … $$ a display block', () => {
    expect(normalizeMath('Formula:\n\n$$x = 1$$\n')).toBe('Formula:\n\n$$\nx = 1\n$$\n')
    expect(normalizeMath('inline $$x$$ stays')).toBe('inline $$x$$ stays')
  })
  it('treats $ before a digit as currency', () => {
    expect(normalizeMath('Budget is $42k but **$30k** here')).toBe('Budget is \\$42k but **\\$30k** here')
    expect(normalizeMath('area $x^2$ and $$2x$$')).toBe('area $x^2$ and $$2x$$')
  })
  it('leaves code alone', () => {
    const src = 'use `\\(x\\)` and\n```\n\\[y\\]\n```'
    expect(normalizeMath(src)).toBe(src)
  })
})

describe('fromSaved', () => {
  it('maps versions and index', () => {
    const m = fromSaved({ role: 'assistant', content: 'b', versions: [[{ role: 'assistant', content: 'a' }]], version_index: 1 }, 3)
    expect(m).toMatchObject({ index: 3, versionCount: 2, versionIndex: 1, id: 'm-3' })
    expect(fromSaved({ role: 'user', content: 'x' }, 0).versionCount).toBeUndefined()
  })
})
