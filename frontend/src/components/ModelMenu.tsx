import { useEffect, useRef, useState } from 'react'
import type { ModelOption } from '../types'
import { CheckIcon, ChevronDownIcon } from './Icons'

interface Props {
  models: ModelOption[]
  selected: string
  onChange: (id: string) => void
  localModelLoaded: boolean
  placement?: 'up' | 'down'
}

const PROVIDER_LABELS: Record<string, string> = { ollama: 'Ollama', lmstudio: 'LM Studio',
  local: 'Local',
  anthropic: 'Anthropic',
  openai: 'OpenAI',
}

export default function ModelMenu({ models, selected, onChange, localModelLoaded, placement = 'up' }: Props) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  const current = models.find(m => m.id === selected)
  const providers = [...new Set(models.map(m => m.provider))]

  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => { if (!ref.current?.contains(e.target as Node)) setOpen(false) }
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false) }
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  return (
    <div className="model-menu" ref={ref}>
      <button className="model-menu-trigger" onClick={() => setOpen(o => !o)} aria-haspopup="listbox" aria-expanded={open}>
        <span>{current?.name ?? 'Select model'}</span>
        <ChevronDownIcon size={14} />
      </button>
      {open && (
        <div className={`popover model-menu-list ${placement}`} role="listbox">
          {providers.map(provider => (
            <div key={provider} className="model-menu-group">
              <div className="model-menu-group-label">{PROVIDER_LABELS[provider] ?? provider}</div>
              {models.filter(m => m.provider === provider).map(m => {
                const loading = m.provider === 'local' && !localModelLoaded
                return (
                  <button
                    key={m.id}
                    role="option"
                    aria-selected={m.id === selected}
                    className="model-menu-item"
                    disabled={loading || !!m.error}
                    title={m.error}
                    onClick={() => { onChange(m.id); setOpen(false) }}
                  >
                    <span className="model-menu-item-name">{m.name}</span>
                    {loading && <span className="model-menu-item-note">loading…</span>}
                    {m.error && <span className="model-menu-item-note">check API key</span>}
                    {m.id === selected && <CheckIcon size={15} className="model-menu-check" />}
                  </button>
                )
              })}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
