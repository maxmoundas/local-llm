import { useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import rehypeHighlight from 'rehype-highlight'
import 'highlight.js/styles/github-dark.css'

interface Props {
  role: 'user' | 'assistant'
  content: string
  /** When true, content is ignored; the component exposes a ref for direct DOM writes. */
  streaming?: boolean
  streamingRef?: React.RefObject<HTMLSpanElement | null>
}

export default function Message({ role, content, streaming, streamingRef }: Props) {
  const endRef = useRef<HTMLDivElement>(null)

  // Auto-scroll when content changes (non-streaming path)
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'instant' })
  }, [content])

  return (
    <div className={`message ${role}`}>
      {role === 'assistant' && <div className="avatar">G</div>}

      <div className="bubble">
        {streaming ? (
          <>
            <span ref={streamingRef} />
            <span className="typing-cursor" />
          </>
        ) : role === 'assistant' ? (
          <ReactMarkdown rehypePlugins={[rehypeHighlight]}>
            {content}
          </ReactMarkdown>
        ) : (
          content
        )}
        <div ref={endRef} />
      </div>
    </div>
  )
}
