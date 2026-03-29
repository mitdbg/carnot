import { useState, useMemo } from 'react'
import { Search, BrainCircuit, FileText, Cog, AlertTriangle, ChevronRight, ChevronDown, Loader2, Database } from 'lucide-react'
import CodeMirror from '@uiw/react-codemirror'
import { python } from '@codemirror/lang-python'
import { vscodeDark } from '@uiw/codemirror-theme-vscode'
import { EditorView } from '@codemirror/view'

/** Maximum characters to show before truncating observations. */
const OBSERVATION_TRUNCATE_LIMIT = 2000

/**
 * Map a planning/execution phase string to a Tailwind colour family.
 */
export function phaseToColor(phase) {
  switch (phase) {
    case 'data_discovery': return 'amber'
    case 'logical_plan':   return 'blue'
    case 'optimizing':     return 'green'
    case 'paraphrase':     return 'purple'
    case 'sem_filter':     return 'teal'
    case 'sem_map':        return 'indigo'
    case 'sem_join':       return 'rose'
    case 'sem_agg':        return 'emerald'
    default:               return 'gray'
  }
}

/** Map a phase to its Lucide icon component. */
function PhaseIcon({ phase, className }) {
  switch (phase) {
    case 'data_discovery': return <Search className={className} />
    case 'logical_plan':   return <BrainCircuit className={className} />
    case 'optimizing':     return <Cog className={className} />
    case 'paraphrase':     return <FileText className={className} />
    default:               return <Cog className={className} />
  }
}

/** Human-friendly label for a phase string. */
function phaseLabel(phase) {
  switch (phase) {
    case 'data_discovery': return 'Data Discovery'
    case 'logical_plan':   return 'Logical Plan'
    case 'optimizing':     return 'Optimizing'
    case 'paraphrase':     return 'Paraphrase'
    default:               return phase?.replace(/_/g, ' ') ?? 'Step'
  }
}

/** Compact CodeMirror theme — read-only, no gutter, small font. */
const compactCodeTheme = EditorView.theme({
  '&': { fontSize: '0.75rem', backgroundColor: '#1e1e1e' },
  '.cm-gutters': { display: 'none' },
  '.cm-content': { padding: '8px' },
  '.cm-scroller': { overflow: 'auto' },
})

/**
 * A collapsible card that visualises a single planning or execution
 * progress event.  Used inside a `step_group` chat message.
 *
 * Representation invariant:
 *   - `phase` is a non-empty string.
 *   - `message` is a non-empty string.
 *   - When `isActive` is true, a spinner is shown on the card.
 *
 * Abstraction function:
 *   Represents one step of agent progress (planning or execution),
 *   rendered as a colour-coded, collapsible card with optional code
 *   and observation detail.
 */
export default function StepCard({
  phase,
  step,
  totalSteps,
  message,
  codeAction,
  observations,
  error,
  stepCostUsd,
  color,
  defaultExpanded = false,
  isActive = false,
  // Execution-specific props
  operatorName,
  operatorIndex,
  totalOperators,
  itemCount,
  previewItems,
  operatorStats,
}) {
  const [expanded, setExpanded] = useState(defaultExpanded)
  const [showFullObservations, setShowFullObservations] = useState(false)

  const colorFamily = color || phaseToColor(phase)
  const hasError = !!error

  // Border / background classes keyed by colour family
  const colorClasses = useMemo(() => {
    if (hasError) return 'border-red-200 bg-red-50'
    const map = {
      amber:   'border-amber-200 bg-amber-50',
      blue:    'border-blue-200 bg-blue-50',
      purple:  'border-purple-200 bg-purple-50',
      teal:    'border-teal-200 bg-teal-50',
      indigo:  'border-indigo-200 bg-indigo-50',
      rose:    'border-rose-200 bg-rose-50',
      emerald: 'border-emerald-200 bg-emerald-50',
      gray:    'border-gray-200 bg-gray-50',
    }
    return map[colorFamily] || map.gray
  }, [colorFamily, hasError])

  // Truncated vs full observations
  const observationsText = observations || ''
  const isTruncated = observationsText.length > OBSERVATION_TRUNCATE_LIMIT
  const displayedObservations = (!showFullObservations && isTruncated)
    ? observationsText.slice(0, OBSERVATION_TRUNCATE_LIMIT) + '…'
    : observationsText

  const hasDetail = codeAction || observationsText || error || previewItems?.length > 0 || operatorStats

  // Format cost: show per-step cost if available, otherwise nothing
  const costDisplay = (typeof stepCostUsd === 'number' && stepCostUsd > 0)
    ? `$${stepCostUsd.toFixed(4)}`
    : null

  return (
    <div className={`border rounded-lg mb-1.5 w-full ${colorClasses} transition-colors`}>
      {/* ── Header (always visible) ── */}
      <button
        onClick={() => hasDetail && setExpanded(prev => !prev)}
        className="w-full flex items-center gap-2 px-3 py-2 text-left text-sm"
      >
        {/* Expand/collapse chevron */}
        {hasDetail ? (
          expanded
            ? <ChevronDown className="w-3.5 h-3.5 text-gray-500 flex-shrink-0" />
            : <ChevronRight className="w-3.5 h-3.5 text-gray-500 flex-shrink-0" />
        ) : (
          <span className="w-3.5 flex-shrink-0" />
        )}

        {/* Phase icon */}
        {hasError
          ? <AlertTriangle className="w-4 h-4 text-red-500 flex-shrink-0" />
          : <PhaseIcon phase={phase} className="w-4 h-4 text-gray-600 flex-shrink-0" />
        }

        {/* Label + step counter */}
        <span className="font-medium text-gray-800 flex-shrink-0">
          {operatorName || phaseLabel(phase)}
          {operatorIndex != null && totalOperators != null && (
            <span className="font-normal text-gray-500"> (step {operatorIndex + 1}/{totalOperators})</span>
          )}
          {operatorIndex == null && step != null && totalSteps != null && (
            <span className="font-normal text-gray-500"> (step {step}/{totalSteps})</span>
          )}
        </span>

        {/* Message (fills available space) */}
        <span className="text-gray-600 truncate flex-1 ml-1">
          {message}
          {itemCount != null && <span className="text-gray-500"> — {itemCount} item{itemCount !== 1 ? 's' : ''}</span>}
        </span>

        {/* Spinner for active step */}
        {isActive && <Loader2 className="w-3.5 h-3.5 animate-spin text-gray-500 flex-shrink-0" />}

        {/* Cost badge */}
        {costDisplay && (
          <span className="text-xs text-gray-500 font-mono flex-shrink-0 ml-2">
            {costDisplay}
          </span>
        )}
      </button>

      {/* ── Expanded detail ── */}
      {expanded && hasDetail && (
        <div className="px-4 pb-3 text-sm border-t border-gray-200/60">
          {/* Error */}
          {error && (
            <div className="mt-2 text-red-700 bg-red-100 rounded px-3 py-2 text-xs font-mono whitespace-pre-wrap">
              {error}
            </div>
          )}

          {/* Code block */}
          {codeAction && (
            <div className="mt-2">
              <div className="text-xs font-medium text-gray-500 mb-1">Code</div>
              <div className="rounded overflow-hidden">
                <CodeMirror
                  value={codeAction}
                  extensions={[python(), compactCodeTheme]}
                  theme={vscodeDark}
                  editable={false}
                  basicSetup={{ lineNumbers: false, foldGutter: false, highlightActiveLine: false }}
                />
              </div>
            </div>
          )}

          {/* Observations */}
          {observationsText && (
            <div className="mt-2">
              <div className="text-xs font-medium text-gray-500 mb-1">Output</div>
              <pre className="bg-gray-100 rounded px-3 py-2 text-xs text-gray-700 whitespace-pre-wrap break-words max-h-64 overflow-auto">
                {displayedObservations}
              </pre>
              {isTruncated && (
                <button
                  onClick={(e) => { e.stopPropagation(); setShowFullObservations(prev => !prev) }}
                  className="text-xs text-blue-600 hover:text-blue-800 mt-1"
                >
                  {showFullObservations ? 'Show less' : 'Show full output'}
                </button>
              )}
            </div>
          )}

          {/* Preview items table (execution) */}
          {previewItems && previewItems.length > 0 && (
            <div className="mt-2">
              <div className="text-xs font-medium text-gray-500 mb-1 flex items-center gap-1">
                <Database className="w-3 h-3" /> Preview
              </div>
              <div className="overflow-auto max-h-48 rounded border border-gray-200">
                <table className="min-w-full text-xs">
                  <thead className="bg-gray-50 sticky top-0">
                    <tr>
                      {Object.keys(previewItems[0]).map(key => (
                        <th key={key} className="px-2 py-1 text-left font-medium text-gray-600 border-b border-gray-200">
                          {key}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {previewItems.map((item, rowIdx) => (
                      <tr key={rowIdx} className={rowIdx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                        {Object.values(item).map((val, colIdx) => (
                          <td key={colIdx} className="px-2 py-1 text-gray-700 border-b border-gray-100 max-w-[200px] truncate">
                            {typeof val === 'object' ? JSON.stringify(val) : String(val ?? '')}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {itemCount != null && itemCount > previewItems.length && (
                <div className="text-xs text-gray-400 mt-1">
                  Showing {previewItems.length} of {itemCount} items
                </div>
              )}
            </div>
          )}

          {/* Operator stats summary (execution) */}
          {operatorStats && (
            <div className="mt-2 flex flex-wrap gap-3 text-xs text-gray-500">
              {operatorStats.items_in != null && (
                <span>{operatorStats.items_in} in → {operatorStats.items_out ?? '?'} out</span>
              )}
              {operatorStats.wall_clock_secs != null && (
                <span>{operatorStats.wall_clock_secs.toFixed(1)}s</span>
              )}
              {operatorStats.total_cost_usd != null && operatorStats.total_cost_usd > 0 && (
                <span>${operatorStats.total_cost_usd.toFixed(4)}</span>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
