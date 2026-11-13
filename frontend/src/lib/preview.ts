import { createContext, useContext } from 'react'
import type { Preview } from '../types'

/** Opens the side preview panel from anywhere (code blocks, file chips, attachments). */
export const PreviewContext = createContext<(p: Preview) => void>(() => {})
export const usePreview = () => useContext(PreviewContext)

const PREVIEWABLE_LANGS = new Set(['html', 'svg', 'xml', 'mermaid'])
export const canPreviewCode = (lang: string) => PREVIEWABLE_LANGS.has(lang.toLowerCase())

export function codePreview(lang: string, code: string): Preview {
  const l = lang.toLowerCase()
  const kind = l === 'mermaid' ? 'mermaid' : l === 'svg' || (l === 'xml' && code.trimStart().startsWith('<svg')) ? 'svg' : 'html'
  return { kind, title: kind === 'mermaid' ? 'Diagram' : kind === 'svg' ? 'SVG' : 'HTML preview', content: code }
}

const TEXT_EXT = /\.(txt|md|markdown|json|jsonl|ya?ml|xml|py|js|ts|tsx|jsx|java|go|rs|rb|php|c|h|cpp|hpp|cs|swift|kt|sql|sh|toml|ini|log|tex|css|r|lua)$/i

/** Decide how to show a downloadable file, by name. */
export function filePreview(name: string, url: string): Preview {
  if (/\.(png|jpe?g|gif|webp|svg)$/i.test(name)) return { kind: 'image', title: name, url }
  return { kind: 'file', title: name, url }
}

export const isPreviewableFile = (name: string) =>
  /\.(png|jpe?g|gif|webp|svg|csv|tsv|html?|md|markdown)$/i.test(name) || TEXT_EXT.test(name)

/** Minimal CSV parser (quoted fields, escaped quotes, CRLF). */
export function parseCsv(text: string, delimiter = ','): string[][] {
  const rows: string[][] = []
  let row: string[] = []
  let field = ''
  let quoted = false
  for (let i = 0; i < text.length; i++) {
    const c = text[i]
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') { field += '"'; i++ }
      else if (c === '"') quoted = false
      else field += c
    } else if (c === '"' && field === '') quoted = true
    else if (c === delimiter) { row.push(field); field = '' }
    else if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i++
      row.push(field); rows.push(row); row = []; field = ''
    } else field += c
  }
  if (field || row.length) { row.push(field); rows.push(row) }
  return rows
}
