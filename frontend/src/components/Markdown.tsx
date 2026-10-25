import { memo, useState, type ReactNode } from 'react'
import ReactMarkdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeHighlight from 'rehype-highlight'
import { CheckIcon, CopyIcon } from './Icons'

function textOf(node: ReactNode): string {
  if (typeof node === 'string' || typeof node === 'number') return String(node)
  if (Array.isArray(node)) return node.map(textOf).join('')
  if (node && typeof node === 'object' && 'props' in node) return textOf((node as { props: { children?: ReactNode } }).props.children)
  return ''
}

function CodeBlock({ children }: { children?: ReactNode }) {
  const [copied, setCopied] = useState(false)
  const code = Array.isArray(children) ? children[0] : children
  const className = (code as { props?: { className?: string } })?.props?.className ?? ''
  const lang = /language-([\w+-]+)/.exec(className)?.[1] ?? ''

  const copy = () => {
    navigator.clipboard.writeText(textOf(children).replace(/\n$/, ''))
    setCopied(true)
    setTimeout(() => setCopied(false), 1600)
  }

  return (
    <div className="code-block">
      <div className="code-block-header">
        <span>{lang}</span>
        <button className="code-copy" onClick={copy} aria-label="Copy code">
          {copied ? <CheckIcon size={14} /> : <CopyIcon size={14} />}
          {copied ? 'Copied' : 'Copy'}
        </button>
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

function Markdown({ content }: { content: string }) {
  return (
    <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[[rehypeHighlight, { detect: true, ignoreMissing: true }]]}
                   components={components}>
      {content}
    </ReactMarkdown>
  )
}

export default memo(Markdown)
