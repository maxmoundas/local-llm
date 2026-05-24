import type { ModelOption } from '../types'

interface Props {
  models: ModelOption[]
  selected: string
  onChange: (id: string) => void
  localModelLoaded: boolean
}

const PROVIDER_LABELS: Record<string, string> = {
  local: 'Local',
  anthropic: 'Anthropic',
  openai: 'OpenAI',
}

export default function ModelSelector({ models, selected, onChange, localModelLoaded }: Props) {
  const providers = [...new Set(models.map(m => m.provider))]

  return (
    <select
      className="model-selector"
      value={selected}
      onChange={e => onChange(e.target.value)}
      title="Select model"
    >
      {providers.map(provider => (
        <optgroup key={provider} label={PROVIDER_LABELS[provider] ?? provider}>
          {models
            .filter(m => m.provider === provider)
            .map(m => {
              const isLocalLoading = m.provider === 'local' && !localModelLoaded
              return (
                <option key={m.id} value={m.id} disabled={isLocalLoading}>
                  {m.name}{isLocalLoading ? ' (loading...)' : ''}
                </option>
              )
            })}
        </optgroup>
      ))}
    </select>
  )
}
