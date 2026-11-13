import type { ModelOption, PendingFile } from '../types'
import Composer from './Composer'
import { SparkIcon } from './Icons'

interface Props {
  displayName: string
  models: ModelOption[]
  modelId: string
  onModelChange: (id: string) => void
  localModelLoaded: boolean
  voiceAvailable: boolean
  onSend: (content: string, files: PendingFile[]) => void
}

function greeting(): string {
  const h = new Date().getHours()
  if (h < 5) return 'Up late'
  if (h < 12) return 'Good morning'
  if (h < 18) return 'Good afternoon'
  return 'Good evening'
}

export default function NewChat({ displayName, models, modelId, onModelChange, localModelLoaded, voiceAvailable, onSend }: Props) {
  const name = displayName.trim()
  return (
    <div className="new-chat-screen">
      <div className="new-chat-inner">
        <h1 className="greeting">
          <SparkIcon size={34} className="greeting-mark" />
          {greeting()}{name ? `, ${name}` : ''}
        </h1>
        <Composer
          onSend={onSend}
          isGenerating={false}
          models={models}
          modelId={modelId}
          onModelChange={onModelChange}
          localModelLoaded={localModelLoaded}
          voiceAvailable={voiceAvailable}
          placeholder="How can I help you today?"
          menuPlacement="down"
        />
      </div>
    </div>
  )
}
