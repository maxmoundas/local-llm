import { useCallback, useEffect, useRef, useState } from 'react'
import { api, formatSize, readFile } from '../api/client'
import type { ModelOption, PendingFile, ProjectDetail } from '../types'
import Composer from './Composer'
import { FileIcon, FolderIcon, PlusIcon, TrashIcon } from './Icons'

interface Props {
  projectId: string
  models: ModelOption[]
  modelId: string
  onModelChange: (id: string) => void
  localModelLoaded: boolean
  voiceAvailable: boolean
  onStartChat: (projectId: string, content: string, files: PendingFile[]) => void
  onOpenChat: (chatId: string) => void
  onChanged: () => void      // project list / names changed
  onDeleted: () => void
}

function relative(iso: string): string {
  const s = (Date.now() - new Date(iso).getTime()) / 1000
  if (s < 60) return 'just now'
  if (s < 3600) return `${Math.floor(s / 60)} min ago`
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`
  return new Date(iso).toLocaleDateString()
}

/** A text field that saves when it loses focus (if changed). */
function useAutosave(initial: string, save: (v: string) => Promise<unknown>) {
  const [value, setValue] = useState(initial)
  const [state, setState] = useState<'idle' | 'saving' | 'saved'>('idle')
  const saved = useRef(initial)
  useEffect(() => { setValue(initial); saved.current = initial }, [initial])
  const commit = async () => {
    if (value === saved.current) return
    setState('saving')
    await save(value)
    saved.current = value
    setState('saved')
    setTimeout(() => setState('idle'), 1500)
  }
  return { value, setValue, commit, state }
}

export default function ProjectView({
  projectId, models, modelId, onModelChange, localModelLoaded, voiceAvailable, onStartChat, onOpenChat, onChanged, onDeleted,
}: Props) {
  const [project, setProject] = useState<ProjectDetail | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [uploading, setUploading] = useState(0)
  const fileInput = useRef<HTMLInputElement>(null)

  const load = useCallback(() => api.getProject(projectId).then(setProject).catch(e => setError(e.message)), [projectId])
  useEffect(() => { setProject(null); load() }, [load])

  const name = useAutosave(project?.name ?? '', async v => { await api.updateProject(projectId, { name: v.trim() || 'Untitled project' }); onChanged() })
  const description = useAutosave(project?.description ?? '', v => api.updateProject(projectId, { description: v }))
  const instructions = useAutosave(project?.instructions ?? '', v => api.updateProject(projectId, { instructions: v }))
  const memory = useAutosave(project?.memory ?? '', v => api.updateProjectMemory(projectId, v))

  const upload = async (list: FileList) => {
    setError(null)
    const picked: PendingFile[] = await Promise.all(Array.from(list).map(readFile))
    setUploading(n => n + picked.length)
    for (const f of picked) {
      try {
        await api.addProjectFile(projectId, f)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Upload failed')
      }
      setUploading(n => n - 1)
    }
    load()
  }

  if (error && !project) return <div className="settings-loading">{error}</div>
  if (!project) return <div className="settings-loading">Loading…</div>

  return (
    <div className="page">
      <div className="page-scroll">
        <div className="page-inner">
          <div className="project-head">
            <FolderIcon size={22} className="project-head-icon" />
            <input className="page-title-input" value={name.value} aria-label="Project name"
                   onChange={e => name.setValue(e.target.value)} onBlur={name.commit}
                   onKeyDown={e => { if (e.key === 'Enter') e.currentTarget.blur() }} />
          </div>
          <input className="page-subtitle-input" value={description.value} placeholder="Add a short description"
                 onChange={e => description.setValue(e.target.value)} onBlur={description.commit}
                 onKeyDown={e => { if (e.key === 'Enter') e.currentTarget.blur() }} aria-label="Project description" />

          <div className="project-composer">
            <Composer
              onSend={(content, files) => onStartChat(projectId, content, files)}
              isGenerating={false}
              models={models}
              modelId={modelId}
              onModelChange={onModelChange}
              localModelLoaded={localModelLoaded}
              voiceAvailable={voiceAvailable}
              placeholder={`Start a chat in ${project.name}…`}
              menuPlacement="down"
            />
          </div>

          <div className="project-grid">
            <section className="project-chats">
              <h2 className="settings-subhead">Chats</h2>
              {project.chats.length === 0 && <p className="muted-text">No chats yet. Start one above.</p>}
              {project.chats.map(c => (
                <button key={c.id} className="project-chat" onClick={() => onOpenChat(c.id)}>
                  <span className="project-chat-name">{c.name}</span>
                  <span className="project-chat-meta">{c.preview || 'Empty'} · {relative(c.updated_at)}</span>
                </button>
              ))}
            </section>

            <aside className="project-side">
              <div className="project-card">
                <div className="project-card-head">
                  <h3>Instructions</h3>
                  {instructions.state !== 'idle' && <span className="saved-hint">{instructions.state === 'saving' ? 'Saving…' : 'Saved'}</span>}
                </div>
                <p className="setting-hint">Sent with every chat in this project.</p>
                <textarea className="text-area" rows={6} value={instructions.value} onBlur={instructions.commit}
                          placeholder="e.g. You're helping me plan a garden. Keep answers short and practical."
                          onChange={e => instructions.setValue(e.target.value)} />
              </div>

              <div className="project-card">
                <div className="project-card-head">
                  <h3>Files</h3>
                  <button className="icon-btn small" onClick={() => fileInput.current?.click()} title="Add files" aria-label="Add files">
                    <PlusIcon size={15} />
                  </button>
                  <input ref={fileInput} type="file" multiple hidden onChange={e => { if (e.target.files?.length) upload(e.target.files); e.target.value = '' }} />
                </div>
                <p className="setting-hint">Documents every chat here can read. Code in the agent's workspace can open them too.</p>
                {project.files.map(f => (
                  <div key={f.id} className="project-file">
                    <FileIcon size={15} />
                    <a href={api.projectFileUrl(projectId, f.id)} download={f.name} className="project-file-name">{f.name}</a>
                    <span className="project-file-meta">{formatSize(f.size)}</span>
                    <button className="icon-btn small" aria-label={`Remove ${f.name}`} title="Remove"
                            onClick={async () => { await api.removeProjectFile(projectId, f.id); load() }}>
                      <TrashIcon size={14} />
                    </button>
                  </div>
                ))}
                {uploading > 0 && <p className="muted-text"><span className="spinner small" /> Reading {uploading} file{uploading === 1 ? '' : 's'}…</p>}
                {error && <p className="form-error">{error}</p>}
              </div>

              <div className="project-card">
                <div className="project-card-head">
                  <h3>Project memory</h3>
                  {memory.state !== 'idle' && <span className="saved-hint">{memory.state === 'saving' ? 'Saving…' : 'Saved'}</span>}
                </div>
                <p className="setting-hint">Facts learned in this project's chats (kept separate from your general memory). Edit freely.</p>
                <textarea className="text-area" rows={5} value={memory.value} onBlur={memory.commit}
                          placeholder="Nothing yet." onChange={e => memory.setValue(e.target.value)} />
              </div>

              <button className="link-btn danger-link" onClick={async () => {
                if (!confirm(`Delete the project "${project.name}"? Its chats are kept and move to Recents.`)) return
                await api.deleteProject(projectId)
                onDeleted()
              }}>Delete project</button>
            </aside>
          </div>
        </div>
      </div>
    </div>
  )
}
