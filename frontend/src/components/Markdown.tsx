import { memo, useState, type ReactNode } from 'react'
import ReactMarkdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeHighlight from 'rehype-highlight'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'
import { canPreviewCode, codePreview, usePreview } from '../lib/preview'
import { CheckIcon, CopyIcon, EyeIcon } from './Icons'

function textOf(node: ReactNode): string {
  if (typeof node === 'string' || typeof node === 'number') return String(node)
  if (Array.isArray(node)) return node.map(textOf).join('')
  if (node && typeof node === 'object' && 'props' in node) return textOf((node as { props: { children?: ReactNode } }).props.children)
  return ''
}

function CodeBlock({ children }: { children?: ReactNode }) {
  const [copied, setCopied] = useState(false)
  const openPreview = usePreview()
  const code = Array.isArray(children) ? children[0] : children
  const className = (code as { props?: { className?: string } })?.props?.className ?? ''
  const lang = /language-([\w+-]+)/.exec(className)?.[1] ?? ''
  const source = () => textOf(children).replace(/\n$/, '')

  const copy = () => {
    navigator.clipboard.writeText(source())
    setCopied(true)
    setTimeout(() => setCopied(false), 1600)
  }

  return (
    <div className="code-block">
      <div className="code-block-header">
        <span>{lang}</span>
        <span className="code-block-actions">
          {canPreviewCode(lang) && (
            <button className="code-copy" onClick={() => openPreview(codePreview(lang, source()))} aria-label="Preview">
              <EyeIcon size={14} /> Preview
            </button>
          )}
          <button className="code-copy" onClick={copy} aria-label="Copy code">
            {copied ? <CheckIcon size={14} /> : <CopyIcon size={14} />}
            {copied ? 'Copied' : 'Copy'}
          </button>
        </span>
      </div>
      <pre>{children}</pre>
    </div>
  )
}

const components: Components = {
  pre: ({ children }) => <CodeBlock>{children}</CodeBlock>,
  a: ({ href, children }) => <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>,
  table: ({ children }) => <div className="table-wrap"><table>{children}</table></div>,
}

/** A `$` right before a digit is usually money ("$42k … $30k") and is escaped, unless it opens inline math
 * like $2x^2 + 5x - 3 = 0$: a closing `$` on the same line, not right after whitespace and not right before a
 * digit (as the next price would be), around math-looking content (no spaces, or an operator). */
function escapeMoney(text: string): string {
  return text.replace(/(?<![\\$])\$(?=(\d[^$\n]*)(\$?))/g, (dollar, body: string, close: string, at: number) => {
    const after = text[at + 1 + body.length + close.length] ?? ''
    const math = close === '$' && !/\s$/.test(body) && !/\d/.test(after)
      && (!/\s/.test(body) || /[\^_\\={}<>+\-*/]/.test(body))
    return math ? dollar : '\\$'
  })
}

/** Models often write \( … \) and \[ … \]; remark-math only knows $ … $ and $$ … $$ (and treats a one-line
 * $$ … $$ as inline). Normalize to what it renders as display math, and escape `$` used for money.
 * Code is left alone. */
export function normalizeMath(src: string): string {
  return src.split(/(```[\s\S]*?(?:```|$)|`[^`\n]*`)/g).map((part, i) => (i % 2 === 1 ? part : escapeMoney(part)
    .replace(/^[ \t]*\$\$([^\n]+?)\$\$[ \t]*$/gm, (_, m) => `$$\n${m.trim()}\n$$`)
    .replace(/\\\[([\s\S]+?)\\\]/g, (_, m) => `\n$$\n${m.trim()}\n$$\n`)
    .replace(/\\\(([\s\S]+?)\\\)/g, (_, m) => `$${m.trim()}$`))).join('')
}

function Markdown({ content }: { content: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, [remarkMath, { singleDollarTextMath: true }]]}
      rehypePlugins={[[rehypeKatex, { throwOnError: false, strict: 'ignore' }], [rehypeHighlight, { detect: true, ignoreMissing: true }]]}
      components={components}
    >
      {normalizeMath(content)}
    </ReactMarkdown>
  )
}

export default memo(Markdown)
