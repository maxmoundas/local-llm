import { useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import rehypeHighlight from 'rehype-highlight'
import 'highlight.js/styles/github-dark.css'

interface Props {
  role: 'user' | 'assistant'
  content: string
  streaming?: boolean
  streamingRef?: React.Ref<HTMLSpanElement>
  imageDataUrl?: string
}

export default function Message({ role, content, streaming, streamingRef, imageDataUrl }: Props) {
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'instant' })
  }, [content])

  return (
    <div className={`message ${role}`}>
      {role === 'assistant' && <div className="avatar">G</div>}

      <div className="bubble">
        {imageDataUrl && role === 'user' && (
          <img
            className="bubble-image"
            src={imageDataUrl}
            alt="attached"
          />
        )}
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
          content && <span>{content}</span>
        )}
        <div ref={endRef} />
      </div>
    </div>
  )
}
