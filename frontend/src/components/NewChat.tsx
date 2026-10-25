import type { AttachedImage, ModelOption } from '../types'
import Composer from './Composer'
import { SparkIcon } from './Icons'

interface Props {
  displayName: string
  models: ModelOption[]
  modelId: string
  onModelChange: (id: string) => void
  localModelLoaded: boolean
  onSend: (content: string, image?: AttachedImage) => void
}

function greeting(): string {
  const h = new Date().getHours()
  if (h < 5) return 'Up late'
  if (h < 12) return 'Good morning'
  if (h < 18) return 'Good afternoon'
  return 'Good evening'
}

export default function NewChat({ displayName, models, modelId, onModelChange, localModelLoaded, onSend }: Props) {
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
          placeholder="How can I help you today?"
          menuPlacement="down"
        />
      </div>
    </div>
  )
}
