import { useState } from 'react'
import type { ResearchStep, SourceRef } from '../types'
import { ChevronDownIcon, CloudIcon, GlobeIcon, NewsIcon, PageIcon } from './Icons'

interface Props {
  steps: ResearchStep[]
  sources: SourceRef[]
  status?: string      // live one-line status while research is running
  active: boolean      // research (or the wait for the first token) is in progress
}

const TOOL_ICON: Record<string, typeof GlobeIcon> = {
  web_search: GlobeIcon,
  news_search: NewsIcon,
  fetch_page: PageIcon,
  get_weather: CloudIcon,
}

function domain(url: string): string {
  try { return new URL(url).hostname.replace(/^www\./, '') } catch { return url }
}

function plural(n: number, word: string): string {
  return `${n} ${word}${n === 1 ? '' : 's'}`
}

const SOURCE_PREVIEW = 8

export default function ResearchSteps({ steps, sources, status, active }: Props) {
  const [open, setOpen] = useState(false)
  const [allSources, setAllSources] = useState(false)
  const shown = allSources ? sources : sources.slice(0, SOURCE_PREVIEW)

  const summary = active
    ? status ?? 'Researching'
    : [plural(steps.length, 'step'), sources.length ? plural(sources.length, 'source') : null]
        .filter(Boolean).join(' · ')

  return (
    <div className={`research${open ? ' open' : ''}${active ? ' active' : ''}`}>
      <button className="research-header" onClick={() => setOpen(o => !o)} aria-expanded={open}>
        {active ? <span className="research-spinner" /> : <GlobeIcon size={15} />}
        <span className={`research-status${active ? ' shimmer' : ''}`} key={active ? summary : 'done'}>
          {active ? summary : `Researched · ${summary}`}
        </span>
        <ChevronDownIcon size={15} className="research-chevron" />
      </button>

      {open && (
        <div className="research-body">
          {steps.length === 0 && <div className="research-empty">Planning…</div>}
          <ol className="research-steps">
            {steps.map((s, i) => {
              const Icon = TOOL_ICON[s.tool] ?? GlobeIcon
              return (
                <li key={i} className="research-step">
                  <Icon size={14} />
                  <span className="research-step-label">{s.label}</span>
                  {s.summary && <span className="research-step-summary">{s.summary}</span>}
                </li>
              )
            })}
          </ol>
          {sources.length > 0 && (
            <div className="research-sources">
              {shown.map(s => (
                <a key={s.url} className="source-chip" href={s.url} target="_blank" rel="noopener noreferrer" title={s.title}>
                  <span className="source-initial">{domain(s.url).charAt(0).toUpperCase()}</span>
                  <span className="source-title">{s.title}</span>
                  <span className="source-domain">{domain(s.url)}</span>
                </a>
              ))}
              {sources.length > SOURCE_PREVIEW && (
                <button className="source-chip more" onClick={() => setAllSources(a => !a)}>
                  {allSources ? 'Show fewer' : `+${sources.length - SOURCE_PREVIEW} more`}
                </button>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
