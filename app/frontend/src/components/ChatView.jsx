import { useRef, useEffect, useCallback } from 'react'
import { Send, Database, AlertCircle, Loader2, XCircle, Play, Code } from 'lucide-react'
import StepCard, { phaseToColor } from './StepCard'
import CostBudgetPicker from './CostBudgetPicker'
import QueryCostDisplay from './QueryCostDisplay'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api"

/**
 * Map an execution operator display name (e.g. "Semantic Filter") to a
 * phase string that `phaseToColor` understands.
 */
function _operatorNameToPhase(operatorName) {
  if (!operatorName) return 'gray'
  const lower = operatorName.toLowerCase()
  if (lower.includes('filter'))   return 'sem_filter'
  if (lower.includes('map'))      return 'sem_map'
  if (lower.includes('join'))     return 'sem_join'
  if (lower.includes('agg'))      return 'sem_agg'
  if (lower.startsWith('dataset')) return 'gray'
  return 'gray'
}

/**
 * The chat message list, plan rendering, and input bar — extracted from
 * UserWorkspacePage.  All shared state is received via props.
 *
 * Props:
 *   messages, isLoading, isExecuting, sessionId, inputQuery, setInputQuery,
 *   costBudget, setCostBudget, queryCostUsd,
 *   onRequestPlan, onExecutePlan, onExecuteInJupyter, onCancel
 */
function ChatView({
  messages,
  isLoading,
  isExecuting,
  sessionId,
  inputQuery,
  setInputQuery,
  costBudget,
  setCostBudget,
  queryCostUsd,
  onRequestPlan,
  onExecutePlan,
  onExecuteInJupyter,
  onCancel,
}) {
  const messagesEndRef = useRef(null)
  const textareaRef = useRef(null)

  const autoResizeTextarea = useCallback(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${el.scrollHeight}px`
  }, [])

  useEffect(() => {
    autoResizeTextarea()
  }, [inputQuery, autoResizeTextarea])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSubmit = (e) => {
    if (e) e.preventDefault()
    onRequestPlan()
  }

  const renderMessage = (message, index) => {
    // Show execute buttons on plan messages unless a result already
    // follows (meaning the plan was executed).  Trailing error messages
    // should NOT hide the buttons — only a 'result' message should.
    const hasResultAfter = messages.slice(index + 1).some(m => m.type === 'result')

    switch (message.type) {
      case 'user':
        return (
          <div key={index} className="flex justify-end mb-4">
            <div className="bg-primary-500 text-white rounded-lg px-4 py-2 max-w-[70%]">
              {message.content}
            </div>
          </div>
        )

      case 'agent':
        return (
          <div key={index} className="flex flex-col items-start mb-4">
            <div className="bg-white border border-gray-200 p-4 rounded-lg max-w-[85%] shadow-sm">
              <div className="prose prose-sm text-gray-700 whitespace-pre-wrap">
                {message.content}
              </div>

              {message.isPlanConfirmation && !hasResultAfter && !isLoading && (
                <div className="mt-4 flex gap-3 border-t pt-4">
                  <button
                    onClick={() => onExecutePlan(message.attachedPlan)}
                    className="flex items-center gap-2 bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-md text-sm font-medium transition-colors"
                  >
                    <Play className="w-4 h-4" /> Execute Plan
                  </button>
                  <button
                    onClick={() => onExecuteInJupyter(message.attachedPlan)}
                    className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-md text-sm font-medium transition-colors"
                  >
                    <Code className="w-4 h-4" /> Execute in Notebook
                  </button>
                </div>
              )}
            </div>
          </div>
        )

      case 'status':
        return (
          <div key={index} className="flex justify-center mb-3">
            <div className="bg-blue-50 text-blue-700 rounded-full px-4 py-1 text-sm flex items-center gap-2">
              <Loader2 className="w-3 h-3 animate-spin" />
              {message.content}
            </div>
          </div>
        )

      case 'error':
        return (
          <div key={index} className="flex justify-center mb-4">
            <div className="bg-red-50 text-red-700 rounded-lg px-4 py-2 flex items-center gap-2">
              <AlertCircle className="w-4 h-4" />
              {message.content}
            </div>
          </div>
        )

      case 'result':
        return (
          <div key={index} className="flex justify-start mb-4">
            <div className="bg-green-50 border border-green-200 rounded-lg p-4 max-w-[80%]">
              <pre className="text-sm text-gray-700 whitespace-pre-wrap mb-3 overflow-auto max-h-[600px]">
                {message.content}
              </pre>
              {message.csv_file && (
                <a
                  href={`${API_BASE_URL}/query/download/${message.csv_file}`}
                  download={message.csv_file}
                  className="inline-flex items-center gap-2 bg-primary-500 hover:bg-primary-600 text-white px-4 py-2 rounded-lg transition-colors"
                >
                  <Database className="w-4 h-4" />
                  Download Full CSV ({message.row_count} rows)
                </a>
              )}
            </div>
          </div>
        )

      case 'step_group':
        return (
          <div key={index} className="flex flex-col items-start mb-4 w-full max-w-[85%]">
            {message.steps.map((step, i) => {
              // Per-step cost is now provided directly by the backend.
              const stepCost = step.step_cost_usd ?? null

              // For execution steps, derive the phase from the operator_name
              // so phaseToColor picks the right colour family.
              const effectivePhase = step.phase || _operatorNameToPhase(step.operator_name)

              return (
                <StepCard
                  key={i}
                  phase={effectivePhase}
                  step={step.step}
                  totalSteps={step.total_steps}
                  message={step.message}
                  codeAction={step.code_action}
                  observations={step.observations}
                  error={step.error}
                  stepCostUsd={stepCost}
                  color={phaseToColor(effectivePhase)}
                  defaultExpanded={i === message.steps.length - 1 && message.isActive}
                  isActive={i === message.steps.length - 1 && message.isActive}
                  operatorName={step.operator_name}
                  operatorIndex={step.operator_index}
                  totalOperators={step.total_operators}
                  itemCount={step.item_count}
                  previewItems={step.preview_items}
                  operatorStats={step.operator_stats}
                />
              )
            })}
          </div>
        )

      default:
        return null
    }
  }

  return (
    <div className="flex-1 flex flex-col min-w-0 bg-white">
      {/* Messages area */}
      <div className="flex-1 overflow-y-auto px-6 py-12">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="w-16 h-16 bg-gray-50 rounded-full flex items-center justify-center mb-6">
              <Database className="w-8 h-8 text-gray-300" />
            </div>
            <h2 className="text-2xl font-semibold text-gray-800 mb-2">Let's get to work.</h2>
            <p className="text-gray-500 max-w-sm">Select a dataset from the left to begin your deep research analysis.</p>
          </div>
        )}

        <div className="max-w-4xl mx-auto w-full">
          {messages.map((message, index) => renderMessage(message, index))}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Sticky Chat Input */}
      <div className="w-full border-t border-gray-100 bg-white px-6 py-4">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-end gap-2">
            <form
              onSubmit={handleSubmit}
              className="relative flex flex-1 items-end gap-2 bg-gray-50 border border-gray-300 rounded-2xl p-2 focus-within:ring-2 focus-within:ring-primary-500 transition-all shadow-sm"
            >
              <textarea
                ref={textareaRef}
                rows="1"
                value={inputQuery}
                onChange={(e) => setInputQuery(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSubmit(); } }}
                placeholder="Ask a question..."
                className="flex-1 bg-transparent border-none focus:ring-0 resize-none py-3 px-4 text-gray-800 overflow-y-auto"
                style={{ maxHeight: '9rem' }}
                disabled={isLoading}
              />
              <div className="flex gap-1 pb-1 pr-1">
                {isLoading && (
                  <button type="button" onClick={onCancel} className="p-2 text-red-500 hover:bg-red-50 rounded-lg">
                    <XCircle className="w-6 h-6" />
                  </button>
                )}
                <button
                  type="submit"
                  className="bg-primary-600 hover:bg-primary-700 text-white p-2 rounded-xl disabled:bg-gray-300 transition-colors"
                  disabled={isLoading || !inputQuery.trim()}
                >
                  {isLoading ? <Loader2 className="w-6 h-6 animate-spin" /> : <Send className="w-6 h-6" />}
                </button>
              </div>
            </form>

            <CostBudgetPicker
              value={costBudget}
              onChange={setCostBudget}
              disabled={isLoading}
            />

            <QueryCostDisplay
              cost={queryCostUsd}
              budget={costBudget}
            />
          </div>
          <p className="text-[10px] text-gray-400 text-center mt-3 uppercase tracking-widest">
            Carnot Research Engine • {sessionId ? `Session: ${sessionId.slice(0, 8)}` : 'Ready'}
          </p>
        </div>
      </div>
    </div>
  )
}

export default ChatView
