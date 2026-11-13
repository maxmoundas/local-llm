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

/** Models often write \( … \) and \[ … \]; remark-math only knows $ … $ and $$ … $$ (and treats a one-line
 * $$ … $$ as inline). Normalize to what it renders as display math. A `$` right before a digit is money
 * ("$42k … $30k"), not math, so it's escaped. Code is left alone. */
export function normalizeMath(src: string): string {
  return src.split(/(```[\s\S]*?(?:```|$)|`[^`\n]*`)/g).map((part, i) => (i % 2 === 1 ? part : part
    .replace(/(?<![\\$])\$(?=\d)/g, '\\$')
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
