import { useEffect, useState } from 'react'
import { parseCsv } from '../lib/preview'
import type { Preview } from '../types'
import { CloseIcon, CopyIcon, DownloadIcon } from './Icons'
import Markdown from './Markdown'

const MAX_ROWS = 500

function CsvTable({ text, delimiter }: { text: string; delimiter: string }) {
  const rows = parseCsv(text, delimiter)
  const [head, ...body] = rows
  if (!head) return <p className="muted-text">Empty file</p>
  return (
    <div className="preview-table-wrap">
      <table className="preview-table">
        <thead><tr>{head.map((h, i) => <th key={i}>{h}</th>)}</tr></thead>
        <tbody>
          {body.slice(0, MAX_ROWS).map((r, i) => <tr key={i}>{r.map((c, j) => <td key={j}>{c}</td>)}</tr>)}
        </tbody>
      </table>
      {body.length > MAX_ROWS && <p className="muted-text preview-note">Showing {MAX_ROWS} of {body.length} rows</p>}
    </div>
  )
}

let mermaidSeq = 0

function MermaidView({ code }: { code: string }) {
  const [svg, setSvg] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    let alive = true
    import('mermaid').then(async ({ default: mermaid }) => {
      const dark = document.documentElement.dataset.theme === 'dark'
        || (!document.documentElement.dataset.theme && matchMedia('(prefers-color-scheme: dark)').matches)
      mermaid.initialize({ startOnLoad: false, securityLevel: 'strict', theme: dark ? 'dark' : 'neutral' })
      try {
        const { svg } = await mermaid.render(`mermaid-${++mermaidSeq}`, code)
        if (alive) setSvg(svg)
      } catch (err) {
        if (alive) setError(err instanceof Error ? err.message : 'Could not render this diagram')
      }
    })
    return () => { alive = false }
  }, [code])
  if (error) return <pre className="preview-error">{error}</pre>
  if (!svg) return <div className="preview-loading"><span className="spinner small" /> Rendering…</div>
  // Rendered with securityLevel "strict" (sanitized, no scripts), then isolated in a sandboxed frame.
  return <iframe className="preview-frame" sandbox="" title="Diagram"
                 srcDoc={`<!doctype html><body style="margin:0;display:grid;place-items:center;min-height:100vh">${svg}</body>`} />
}

/** Fetch a file and show it according to its type. */
function FileView({ title, url }: { title: string; url: string }) {
  const [text, setText] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    let alive = true
    fetch(url).then(async r => {
      if (!r.ok) throw new Error(`Couldn't load the file (${r.status})`)
      const blob = await r.blob()
      if (blob.size > 5 * 1024 * 1024) throw new Error('Too large to preview — download it instead.')
      return blob.text()
    }).then(t => alive && setText(t)).catch(err => alive && setError(err.message))
    return () => { alive = false }
  }, [url])
  if (error) return <p className="preview-error">{error}</p>
  if (text === null) return <div className="preview-loading"><span className="spinner small" /> Loading…</div>
  if (/\.csv$/i.test(title)) return <CsvTable text={text} delimiter="," />
  if (/\.tsv$/i.test(title)) return <CsvTable text={text} delimiter={'\t'} />
  if (/\.html?$/i.test(title)) return <iframe className="preview-frame" sandbox="allow-scripts" srcDoc={text} title={title} />
  if (/\.(md|markdown)$/i.test(title)) return <div className="prose preview-prose"><Markdown content={text} /></div>
  return <pre className="preview-code">{text}</pre>
}

export default function PreviewPanel({ preview, onClose }: { preview: Preview; onClose: () => void }) {
  const [copied, setCopied] = useState(false)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const source = 'content' in preview ? preview.content : null
  return (
    <aside className="preview-panel" aria-label="Preview">
      <header className="preview-header">
        <span className="preview-title">{preview.title}</span>
        {source !== null && (
          <button className="icon-btn small" title={copied ? 'Copied' : 'Copy source'} aria-label="Copy source"
                  onClick={() => { navigator.clipboard.writeText(source); setCopied(true); setTimeout(() => setCopied(false), 1500) }}>
            <CopyIcon size={15} />
          </button>
        )}
        {'url' in preview && (
          <a className="icon-btn small" href={preview.url} download={preview.title} title="Download" aria-label="Download">
            <DownloadIcon size={15} />
          </a>
        )}
        <button className="icon-btn small" onClick={onClose} title="Close (Esc)" aria-label="Close preview"><CloseIcon size={15} /></button>
      </header>
      <div className="preview-body">
        {preview.kind === 'html' && (
          // Scripts may run, but the frame has no same-origin access: it can't read the app or call the API.
          <iframe className="preview-frame" sandbox="allow-scripts allow-modals" srcDoc={preview.content} title="HTML preview" />
        )}
        {preview.kind === 'svg' && (
          <iframe className="preview-frame" sandbox="" title="SVG preview"
                  srcDoc={`<!doctype html><body style="margin:0;display:grid;place-items:center;min-height:100vh">${preview.content}</body>`} />
        )}
        {preview.kind === 'mermaid' && <MermaidView code={preview.content} />}
        {preview.kind === 'markdown' && <div className="prose preview-prose"><Markdown content={preview.content} /></div>}
        {preview.kind === 'text' && <pre className="preview-code">{preview.content}</pre>}
        {preview.kind === 'csv' && <CsvTable text={preview.content} delimiter="," />}
        {preview.kind === 'image' && <div className="preview-image"><img src={preview.url} alt={preview.title} /></div>}
        {preview.kind === 'file' && <FileView title={preview.title} url={preview.url} />}
      </div>
    </aside>
  )
}
