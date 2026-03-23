import { useState, useEffect, useRef } from 'react'
import { Database, CheckSquare, Square, MessageSquare, Trash2, ChevronLeft, ChevronRight, Search, Plus, X } from 'lucide-react'
import { useApiToken } from '../hooks/useApiToken'
import { useQueryEventsPolling } from '../hooks/useQueryEventsPolling'
import { conversationsApi, datasetsApi, notebookApi, workspacesApi } from '../services/api'
import TabBar from '../components/TabBar'
import ChatView from '../components/ChatView'
import NotebookView from '../components/NotebookView'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api"
const DEFAULT_COST_BUDGET = 5.00

/* ─── Plan Visualizer (unchanged) ──────────────────────────────────── */

function PlanVisualizer({ plan }) {
  const flattenPlan = (node, acc = new Map()) => {
    if (!node) return acc
    if (node.parents && node.parents.length > 0) {
      node.parents.forEach(parent => flattenPlan(parent, acc))
    }
    const nodeKey = JSON.stringify({ name: node.name, params: node.params })
    if (!acc.has(nodeKey)) {
      acc.set(nodeKey, {
        name: node.name,
        operator: node.params?.operator || 'Source',
        description: node.params?.description || `Load dataset: ${node.name}`,
        parents: node.parents?.map(p => p.name) || [],
        details: node.params || {}
      })
    }
    return acc
  }

  const stepsMap = plan ? flattenPlan(plan) : new Map()
  const steps = Array.from(stepsMap.values())

  if (steps.length === 0) return (
    <div className="flex flex-col items-center justify-center h-full text-gray-400 p-4 text-center">
      <p>No active execution plan.</p>
    </div>
  )

  return (
    <div className="p-4 overflow-y-auto h-full">
      <div className="space-y-0">
        {steps.map((step, idx) => {
          const isCombination = step.parents.length > 1
          return (
            <div key={idx} className="flex gap-4">
              <div className="flex flex-col items-center">
                <div className={`w-3 h-3 rounded-full border-2 transition-colors ${
                  idx === steps.length - 1
                    ? 'bg-primary-500 border-primary-500'
                    : 'bg-white border-gray-300'
                }`} />
                {idx !== steps.length - 1 && <div className="w-0.5 h-full bg-gray-200" />}
              </div>
              <div className="flex-1 pb-8">
                <div className={`bg-white p-3 rounded-lg border shadow-sm transition-all ${
                  isCombination ? 'border-amber-200 bg-amber-50/30' : 'border-gray-200'
                }`}>
                  <div className="flex items-center justify-between mb-2">
                    <span className={`text-[10px] font-bold px-2 py-0.5 rounded uppercase font-mono ${
                      isCombination
                        ? 'bg-amber-100 text-amber-700'
                        : 'bg-primary-50 text-primary-700'
                    }`}>
                      {step.operator}
                    </span>
                    <span className="text-[10px] text-gray-400 font-mono">Node {idx + 1}</span>
                  </div>
                  <p className="text-sm text-gray-800 leading-snug">{step.description}</p>
                  {isCombination && (
                    <div className="mt-3 pt-2 border-t border-amber-100">
                      <p className="text-[10px] font-bold text-amber-600 uppercase mb-1">Combining Inputs:</p>
                      <div className="flex flex-wrap gap-1">
                        {step.parents.map((p, pIdx) => (
                          <span key={pIdx} className="text-[9px] bg-white border border-amber-200 px-1.5 py-0.5 rounded text-gray-600 italic">
                            {p}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  {step.details.condition && (
                    <div className="mt-2 text-[11px] bg-gray-50 p-1.5 rounded border border-gray-100 font-mono text-gray-600 break-all">
                      <span className="text-blue-600 font-bold">WHERE:</span> {step.details.condition}
                    </div>
                  )}
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

/* ─── Workspace Page ───────────────────────────────────────────────── */

function UserWorkspacePage() {
  const getValidToken = useApiToken()

  // ─── Shared state (used by sidebar + chat) ────────────────────────
  const [datasets, setDatasets] = useState([])
  const [selectedDatasets, setSelectedDatasets] = useState(new Set())
  const [messages, setMessages] = useState([])
  const [inputQuery, setInputQuery] = useState('')
  const [isExecuting, setIsExecuting] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [sessionId, setSessionId] = useState(null)
  const [workspaces, setWorkspaces] = useState([])
  const [currentWorkspaceId, setCurrentWorkspaceId] = useState(null)
  const [currentConversationId, setCurrentConversationId] = useState(null)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [datasetSearchQuery, setDatasetSearchQuery] = useState('')
  const [currentPlan, setCurrentPlan] = useState(null)
  const [lastQuery, setLastQuery] = useState('')
  const [costBudget, setCostBudget] = useState(DEFAULT_COST_BUDGET)
  const [queryCostUsd, setQueryCostUsd] = useState(null)
  const abortControllerRef = useRef(null)
  const currentWorkspaceIdRef = useRef(null)

  // ─── Workspace tab state ──────────────────────────────────────────
  const [tabs, setTabs] = useState([
    { id: 'chat', label: 'Chat', type: 'chat' }
  ])
  const [activeTabId, setActiveTabId] = useState('chat')
  const [notebooks, setNotebooks] = useState({})   // { [notebookId]: { notebookId, query, cells } }
  const notebookCountRef = useRef(0)
  // Track the plan visualizer state — auto-dismiss on notebook switch
  const [showPlanVisualizer, setShowPlanVisualizer] = useState(true)

  // ─── Poll-based catch-up for in-flight queries (Phase 5) ──────────
  // When the user navigates back to a workspace whose query is still
  // running, we poll GET /query/events to reconstruct the streaming UI.
  // shouldPoll is only true when an active query was detected AND no
  // SSE stream is connected (to avoid double-counting events).
  const [queryActiveOnLoad, setQueryActiveOnLoad] = useState(false)
  const shouldPoll = queryActiveOnLoad && !isLoading && !isExecuting

  const { events: polledEvents, isComplete: pollIsComplete, reset: resetPolling } =
    useQueryEventsPolling(currentConversationId, shouldPoll, getValidToken)

  // ─── Lifecycle ────────────────────────────────────────────────────
  useEffect(() => {
    loadWorkspaces()
  }, [])

  useEffect(() => { loadDatasets() }, [])

  // ─── Dispatch polled events into the same state as SSE (Phase 5) ──
  // polledEvents is an ever-growing array; we track how many we've
  // already processed so we only dispatch new ones.
  const processedPollCountRef = useRef(0)

  useEffect(() => {
    if (polledEvents.length <= processedPollCountRef.current) return

    const newEvents = polledEvents.slice(processedPollCountRef.current)
    processedPollCountRef.current = polledEvents.length

    for (const event of newEvents) {
      const data = event.payload

      if (event.event_type === 'step_detail') {
        const source = event.source || data.source
        if (event.step_cost_usd != null) {
          setQueryCostUsd(prev => (prev || 0) + event.step_cost_usd)
        }
        setMessages(prev => {
          const last = prev[prev.length - 1]
          if (last?.type === 'step_group' && last.source === source) {
            const updated = { ...last, steps: [...last.steps, data] }
            return [...prev.slice(0, -1), updated]
          }
          return [...prev, { type: 'step_group', source, steps: [data], isActive: true }]
        })
      } else if (event.event_type === 'result') {
        setMessages(prev => [
          ...prev.map(m =>
            m.type === 'step_group' && m.isActive ? { ...m, isActive: false } : m
          ),
          {
            type: 'result',
            content: data.message,
            csv_file: data.csv_file,
            row_count: data.row_count,
          },
        ])
        loadWorkspaces()
        setCurrentPlan(null)
      } else if (event.event_type === 'error') {
        setMessages(prev => [
          ...prev.map(m =>
            m.type === 'step_group' && m.isActive ? { ...m, isActive: false } : m
          ),
          { type: 'error', content: data.message },
        ])
      } else if (event.event_type === 'execution_stats') {
        if (data.total_cost_usd != null) {
          setQueryCostUsd(data.total_cost_usd)
        }
      } else if (event.event_type === 'planning_stats') {
        if (data.total_cost_usd != null) {
          setQueryCostUsd(data.total_cost_usd)
        }
      } else if (event.event_type === 'plan_complete') {
        setMessages(prev => {
          const planMsg = {
            type: 'agent',
            content: data.natural_language_plan,
            isPlanConfirmation: true,
            attachedPlan: data.plan,
          }
          return [
            ...prev.map(m =>
              m.type === 'step_group' && m.isActive ? { ...m, isActive: false } : m
            ),
            planMsg,
          ]
        })
        if (data.plan) setCurrentPlan(data.plan)
      } else if (event.event_type === 'done') {
        setMessages(prev =>
          prev.map(m =>
            m.type === 'step_group' && m.isActive ? { ...m, isActive: false } : m
          )
        )
      }
    }
  }, [polledEvents])

  // When polling detects the query finished, turn off the active flag
  useEffect(() => {
    if (pollIsComplete && queryActiveOnLoad) {
      setQueryActiveOnLoad(false)
    }
  }, [pollIsComplete, queryActiveOnLoad])

  // ─── Data loaders ──────────────────────────────────────────────────
  const loadWorkspaces = async () => {
    try {
      const token = await getValidToken()
      if (!token) return
      const response = await workspacesApi.list(token)
      setWorkspaces(response.data)
    } catch (error) {
      console.error('Error loading workspaces:', error)
    }
  }

  const loadWorkspace = async (workspaceId) => {
    // Update the ref synchronously (before any await) so inflight SSE
    // handlers can detect the switch immediately.
    currentWorkspaceIdRef.current = workspaceId

    // Abort any inflight SSE stream from a previous workspace.
    // This is safe because the backend persists events independently
    // of the SSE consumer — no data is lost.
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null
    }
    setIsLoading(false)
    setIsExecuting(false)
    // Reset polling state from any previous workspace
    resetPolling()
    processedPollCountRef.current = 0
    setQueryActiveOnLoad(false)

    try {
      const token = await getValidToken()
      if (!token) return
      const wsResponse = await workspacesApi.get(workspaceId, token)
      const workspace = wsResponse.data

      setCurrentWorkspaceId(workspace.id)

      // Restore dataset selection from workspace.
      // Guard against empty strings (falsy in JS) and malformed values.
      if (workspace.dataset_ids && workspace.dataset_ids.trim()) {
        const datasetIds = workspace.dataset_ids.split(',').map(id => parseInt(id)).filter(id => !isNaN(id))
        setSelectedDatasets(new Set(datasetIds))
      } else {
        setSelectedDatasets(new Set())
      }

      // Load the first (and currently only) conversation's messages
      const firstConv = workspace.conversations?.[0]
      if (firstConv) {
        const convResponse = await conversationsApi.get(firstConv.id, token)
        const conversation = convResponse.data

        setSessionId(conversation.session_id)
        setCurrentConversationId(conversation.id)

        // Reconstruct messages, restoring plan confirmation buttons.
        // The backend stores two adjacent messages for a plan:
        //   1. type="natural-language-plan" — the NL description
        //   2. type="logical-plan"          — the JSON logical plan
        // We pair them so the NL message gets isPlanConfirmation + attachedPlan,
        // and we skip the raw logical-plan message from the visible list.
        const dbMessages = conversation.messages
        let restoredPlan = null
        let restoredLastQuery = ''
        const formattedMessages = []

        for (let i = 0; i < dbMessages.length; i++) {
          const msg = dbMessages[i]

          // Skip raw logical-plan messages — they're consumed by the
          // natural-language-plan handler below.
          if (msg.type === 'logical-plan') continue

          // Restore persisted step_group messages as collapsed card groups.
          if (msg.type === 'step_group') {
            try {
              const steps = JSON.parse(msg.content)
              formattedMessages.push({
                type: 'step_group',
                source: steps[0]?.source || 'planning',
                steps,
                isActive: false,
              })
            } catch (_) { /* malformed JSON — skip */ }
            continue
          }

          // Track the last user query so execute handlers can use it
          if (msg.role === 'user') {
            restoredLastQuery = msg.content
          }

          if (msg.type === 'natural-language-plan') {
            // Look ahead for the paired logical-plan message
            let attachedPlan = null
            if (i + 1 < dbMessages.length && dbMessages[i + 1].type === 'logical-plan') {
              try {
                attachedPlan = JSON.parse(dbMessages[i + 1].content)
                restoredPlan = attachedPlan
              } catch (_) { /* malformed JSON — leave attachedPlan null */ }
            }
            formattedMessages.push({
              type: msg.role,
              content: msg.content,
              isPlanConfirmation: true,
              attachedPlan,
            })
          } else {
            // Use msg.type for known display types (result, error) so
            // ChatView renders them with the correct styling (green bg,
            // CSV download button, etc.) rather than as plain agent bubbles.
            formattedMessages.push({
              type: (msg.type === 'result' || msg.type === 'error') ? msg.type : msg.role,
              content: msg.content,
              csv_file: msg.csv_file,
              row_count: msg.row_count,
            })
          }
        }

        setMessages(formattedMessages)
        setCurrentPlan(restoredPlan)
        setLastQuery(restoredLastQuery)
      } else {
        // Empty workspace — no conversation yet
        setCurrentConversationId(null)
        setSessionId(workspace.session_id)
        setMessages([])
        setCurrentPlan(null)
        setLastQuery('')
      }

      setQueryCostUsd(workspace.total_cost_usd || null)

      // If any conversation has an active in-flight query, enable
      // poll-based catch-up so the streaming UI reconstructs.
      const activeConv = (workspace.conversations || []).find(c => c.is_query_active)
      if (activeConv) {
        setQueryActiveOnLoad(true)
      }

      // Switch to chat tab when loading a workspace
      setActiveTabId('chat')
      setShowPlanVisualizer(true)

      // ─── Restore notebook tabs from workspace.notebooks ──
      const wsNotebooks = workspace.notebooks || []
      if (wsNotebooks.length > 0) {
        const restoredNotebooks = {}
        const restoredTabs = [{ id: 'chat', label: 'Chat', type: 'chat' }]

        for (const nbSummary of wsNotebooks) {
          try {
            const nbResp = await notebookApi.get(nbSummary.notebook_uuid, token)
            const nbData = nbResp.data
            const cellsWithOutput = (nbData.cells || []).filter(c => c.output).length
            console.log(
              `[loadWorkspace] notebook=${nbSummary.notebook_uuid} active=${nbData.active}` +
              ` total_cells=${(nbData.cells || []).length} cells_with_output=${cellsWithOutput}`
            )
            restoredNotebooks[nbSummary.notebook_uuid] = {
              notebookId: nbSummary.notebook_uuid,
              query: nbData.query,
              cells: nbData.cells || [],
              active: nbData.active,  // whether in-memory state is alive
            }
            restoredTabs.push({
              id: nbSummary.notebook_uuid,
              label: nbSummary.label || `Notebook ${restoredTabs.length}`,
              type: 'notebook',
            })
          } catch (err) {
            console.warn('Failed to restore notebook', nbSummary.notebook_uuid, err)
          }
        }

        setNotebooks(restoredNotebooks)
        setTabs(restoredTabs)
        notebookCountRef.current = wsNotebooks.length
      } else {
        // Target workspace has no notebooks — clear local notebook state.
        // Do NOT call notebookApi.close() here: that would delete the
        // DB rows of the *previous* workspace's notebooks.  The backend's
        // _cleanup_old_notebooks() timeout handles in-memory eviction.
        setNotebooks({})
        setTabs([{ id: 'chat', label: 'Chat', type: 'chat' }])
        notebookCountRef.current = 0
      }
    } catch (error) {
      console.error('Error loading workspace:', error)
    }
  }

  const deleteWorkspace = async (workspaceId, e) => {
    e.stopPropagation()
    if (!window.confirm('Are you sure you want to delete this workspace?')) return
    try {
      const token = await getValidToken()
      if (!token) return
      await workspacesApi.delete(workspaceId, token)
      if (workspaceId === currentWorkspaceId) {
        // Reset to empty state
        setCurrentWorkspaceId(null)
        setCurrentConversationId(null)
        setSessionId(null)
        setMessages([])
        setSelectedDatasets(new Set())
        setCurrentPlan(null)
        setLastQuery('')
        setQueryCostUsd(null)
        setNotebooks({})
        setTabs([{ id: 'chat', label: 'Chat', type: 'chat' }])
        setActiveTabId('chat')
        notebookCountRef.current = 0
        setShowPlanVisualizer(true)
      }
      loadWorkspaces()
    } catch (error) {
      console.error('Error deleting workspace:', error)
    }
  }

  const handleNewWorkspace = async () => {
    // Guard: if the current workspace is already empty, just reset input state
    if (messages.length === 0 && Object.keys(notebooks).length === 0 && currentWorkspaceId) {
      setInputQuery('')
      return
    }

    // Guard: if there's already an unused workspace (one with no messages),
    // navigate to it instead of creating another.
    const existingEmpty = workspaces.find(ws => ws.message_count === 0)
    if (existingEmpty) {
      loadWorkspace(existingEmpty.id)
      return
    }

    try {
      const token = await getValidToken()
      if (!token) return

      // Backend generates session_id for both workspace and its first conversation
      const response = await workspacesApi.create({
        title: 'Untitled Workspace',
      }, token)

      const workspace = response.data
      const firstConversation = workspace.conversations[0]

      setCurrentWorkspaceId(workspace.id)
      setCurrentConversationId(firstConversation.id)
      setSessionId(firstConversation.session_id)
      setMessages([])
      setInputQuery('')
      setCurrentPlan(null)
      setLastQuery('')
      setSelectedDatasets(new Set())
      setQueryCostUsd(null)
      // Reset workspace: clear local notebook state (don't delete DB rows)
      setNotebooks({})
      setTabs([{ id: 'chat', label: 'Chat', type: 'chat' }])
      setActiveTabId('chat')
      notebookCountRef.current = 0
      setShowPlanVisualizer(true)

      loadWorkspaces()  // refresh sidebar
    } catch (error) {
      console.error('Error creating workspace:', error)
    }
  }

  const loadDatasets = async () => {
    try {
      const token = await getValidToken()
      if (!token) return
      const response = await datasetsApi.list(token)
      setDatasets(response.data)
    } catch (error) {
      console.error('Error loading datasets:', error)
    }
  }

  const toggleDataset = (datasetId) => {
    // Capture the workspace ID synchronously so the async persist uses
    // the correct value even if React batches a state update.
    const wsId = currentWorkspaceId
    setSelectedDatasets(prev => {
      const newSet = new Set(prev)
      if (newSet.has(datasetId)) newSet.delete(datasetId)
      else newSet.add(datasetId)

      // Persist dataset selection to workspace if one exists
      if (wsId) {
        const updatedIds = Array.from(newSet).map(id => parseInt(id)).join(',')
        getValidToken().then(token => {
          if (token) workspacesApi.update(wsId, { dataset_ids: updatedIds }, token).catch(console.error)
        })
      }

      return newSet
    })
  }

  const filteredDatasets = datasets.filter(dataset => {
    if (!datasetSearchQuery.trim()) return true
    const q = datasetSearchQuery.toLowerCase()
    return dataset.name.toLowerCase().includes(q) || dataset.annotation.toLowerCase().includes(q)
  })

  // ─── Chat handlers ────────────────────────────────────────────────

  /** Remove trailing client-side error messages (e.g. "select a dataset",
   *  "API key missing") so they don't persist after the user corrects the
   *  problem and resubmits. */
  const stripTrailingErrors = () => {
    setMessages(prev => {
      let end = prev.length
      while (end > 0 && prev[end - 1].type === 'error') end--
      return end === prev.length ? prev : prev.slice(0, end)
    })
  }

  const handleCancel = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      setMessages(prev => [...prev, { type: 'error', content: 'Query cancelled by user.' }])
      setIsLoading(false)
      abortControllerRef.current = null
    }
  }

  const handleRequestPlan = async () => {
    if (!inputQuery.trim()) return

    // Clear any lingering client-side error messages before proceeding
    stripTrailingErrors()

    if (selectedDatasets.size === 0) {
      setMessages(prev => [...prev, { type: 'error', content: 'Please select at least one dataset before submitting a query.' }])
      return
    }

    // Snapshot dataset selection as a string before any async work —
    // this avoids any stale-closure issues with React state.
    const datasetIdsStr = Array.from(selectedDatasets).map(id => parseInt(id)).join(',')

    // Ensure a workspace + conversation exist before submitting
    let currentSessionId = sessionId
    let workspaceId = currentWorkspaceId
    if (!currentSessionId) {
      try {
        const token = await getValidToken()
        if (!token) return
        const response = await workspacesApi.create({ title: 'Untitled Workspace', dataset_ids: datasetIdsStr }, token)
        const workspace = response.data
        const firstConv = workspace.conversations[0]
        workspaceId = workspace.id
        setCurrentWorkspaceId(workspace.id)
        setCurrentConversationId(firstConv.id)
        setSessionId(firstConv.session_id)
        currentSessionId = firstConv.session_id
        loadWorkspaces()
      } catch (error) {
        console.error('Error auto-creating workspace:', error)
        setMessages(prev => [...prev, { type: 'error', content: 'Failed to create workspace. Please try again.' }])
        return
      }
    }

    // Always persist dataset selection to workspace — this is a safety net
    // in case the create above didn't carry the value, or the workspace
    // already existed but with stale dataset_ids.
    if (workspaceId) {
      try {
        const token = await getValidToken()
        if (token) {
          await workspacesApi.update(workspaceId, { dataset_ids: datasetIdsStr }, token)
        }
      } catch (error) {
        console.warn('Failed to persist dataset selection:', error)
      }
    }

    const queryToPlan = inputQuery
    setLastQuery(queryToPlan)

    setMessages(prev => [...prev, { type: 'user', content: queryToPlan }])
    setInputQuery('')
    setIsLoading(true)
    setIsExecuting(false)
    setQueryCostUsd(null)

    let titleRefreshed = false

    try {
      const token = await getValidToken()
      if (!token) return

      abortControllerRef.current = new AbortController()
      const workspaceAtStart = currentWorkspaceIdRef.current
      const datasetIds = Array.from(selectedDatasets).map(id => parseInt(id))

      const response = await fetch(`${API_BASE_URL}/query/plan`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
        body: JSON.stringify({
          query: queryToPlan,
          dataset_ids: datasetIds,
          session_id: currentSessionId,
          plan: currentPlan,
          cost_budget: costBudget
        }),
        signal: abortControllerRef.current.signal
      })

      if (!response.ok) {
        let errorData
        try { errorData = await response.json() } catch (_) {
          throw new Error(`Server responded with ${response.status}: ${response.statusText}`)
        }
        throw errorData.detail || new Error(errorData.message || 'An unknown server error occurred.')
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            // Phase 1: stop processing if the user navigated away
            if (currentWorkspaceIdRef.current !== workspaceAtStart) break
            try {
              const data = JSON.parse(line.slice(6))
              if (data.session_id && data.session_id !== sessionId) setSessionId(data.session_id)

              if (data.type === 'step_detail' && data.source === 'planning') {
                if (data.step_cost_usd != null) setQueryCostUsd(prev => (prev || 0) + data.step_cost_usd)
                // Refresh sidebar on the first status event so the
                // workspace title (set by the backend immediately after
                // receiving the user message) appears without waiting
                // for plan_complete.
                if (!titleRefreshed) { loadWorkspaces(); titleRefreshed = true }

                setMessages(prev => {
                  const last = prev[prev.length - 1]
                  if (last?.type === 'step_group' && last.source === 'planning') {
                    // Append to existing step group
                    const updated = { ...last, steps: [...last.steps, data] }
                    return [...prev.slice(0, -1), updated]
                  } else {
                    // Start a new step group
                    return [...prev, { type: 'step_group', source: 'planning', steps: [data], isActive: true }]
                  }
                })
              } else if (data.type === 'planning_stats') {
                if (data.total_cost_usd != null) setQueryCostUsd(data.total_cost_usd)
              } else if (data.type === 'plan_complete') {
                const planMsg = {
                  type: 'agent',
                  content: data.natural_language_plan,
                  isPlanConfirmation: true,
                  attachedPlan: data.plan
                }
                // Mark the step group as no longer active and append the plan message
                setMessages(prev => {
                  const updated = prev.map(m =>
                    m.type === 'step_group' && m.isActive ? { ...m, isActive: false } : m
                  )
                  return [...updated, planMsg]
                })
                setCurrentPlan(data.plan)
                loadWorkspaces()  // refresh sidebar — title derived from query
              } else if (data.type === 'error') {
                const errMsg = { type: 'error', content: data.message }
                // Mark the step group as no longer active and append the error
                setMessages(prev => {
                  const updated = prev.map(m =>
                    m.type === 'step_group' && m.isActive ? { ...m, isActive: false } : m
                  )
                  return [...updated, errMsg]
                })
              }
            } catch (e) {
              console.error('Error parsing SSE data:', e)
            }
          }
        }
      }
    } catch (error) {
      // Mark any active step group as no longer active
      setMessages(prev => prev.map(m =>
        m.type === 'step_group' && m.isActive ? { ...m, isActive: false } : m
      ))
      if (error && error.type === 'API_KEY_MISSING') {
        setMessages(prev => [...prev, { type: 'error', content: `${error.message} Please go to the Settings page to configure your keys.` }])
      } else if (error.name !== 'AbortError') {
        console.error('Error planning query:', error)
        setMessages(prev => [...prev, { type: 'error', content: error?.message || 'Failed to generate plan. Please try again.' }])
      }
    } finally {
      setIsLoading(false)
      abortControllerRef.current = null
    }
  }

  const handleExecutePlan = async (planToUse) => {
    const plan = planToUse || currentPlan
    setIsLoading(true)
    setIsExecuting(true)

    // Clear any lingering client-side error messages
    stripTrailingErrors()

    // Prefer lastQuery; fall back to the last user message in the chat
    const queryText = lastQuery || messages.findLast(m => m.type === 'user')?.content || ''
    if (!queryText) { setIsLoading(false); return }
    if (selectedDatasets.size === 0) {
      setMessages(prev => [...prev, { type: 'error', content: 'Please select at least one dataset before submitting a query.' }])
      setIsLoading(false)
      return
    }

    // Ensure a workspace (and session) exists before executing
    let currentSessionId = sessionId
    if (!currentWorkspaceId || !currentSessionId) {
      try {
        const token = await getValidToken()
        if (!token) { setIsLoading(false); return }
        const datasetIds = Array.from(selectedDatasets).map(id => parseInt(id))
        const wsResp = await workspacesApi.create({ title: 'Untitled Workspace', dataset_ids: datasetIds.join(',') }, token)
        const ws = wsResp.data
        setCurrentWorkspaceId(ws.id)
        const firstConv = ws.conversations?.[0]
        if (firstConv) {
          setCurrentConversationId(firstConv.id)
          currentSessionId = firstConv.session_id
          setSessionId(currentSessionId)
        }
        loadWorkspaces()
      } catch (err) {
        console.error('Error auto-creating workspace:', err)
        setIsLoading(false)
        return
      }
    }

    try {
      const token = await getValidToken()
      if (!token) return

      abortControllerRef.current = new AbortController()
      const workspaceAtStart = currentWorkspaceIdRef.current
      const datasetIds = Array.from(selectedDatasets).map(id => parseInt(id))

      const response = await fetch(`${API_BASE_URL}/query/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
        body: JSON.stringify({
          query: queryText,
          dataset_ids: datasetIds,
          session_id: currentSessionId,
          plan: plan,
          cost_budget: costBudget
        }),
        signal: abortControllerRef.current.signal
      })

      if (!response.ok) {
        let errorData
        try { errorData = await response.json() } catch (_) {
          throw new Error(`Server responded with ${response.status}: ${response.statusText}`)
        }
        throw errorData.detail || new Error(errorData.message || 'An unknown server error occurred.')
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            // stop processing if the user navigated away
            if (currentWorkspaceIdRef.current !== workspaceAtStart) break
            try {
              const data = JSON.parse(line.slice(6))
              if (data.session_id && data.session_id !== sessionId) setSessionId(data.session_id)

              if (data.type === 'step_detail' && data.source === 'execution') {
                if (data.step_cost_usd != null) setQueryCostUsd(prev => (prev || 0) + data.step_cost_usd)
                setMessages(prev => {
                  const last = prev[prev.length - 1]
                  if (last?.type === 'step_group' && last.source === 'execution') {
                    const updated = { ...last, steps: [...last.steps, data] }
                    return [...prev.slice(0, -1), updated]
                  } else {
                    return [...prev, { type: 'step_group', source: 'execution', steps: [data], isActive: true }]
                  }
                })
              } else if (data.type === 'result') {
                // Mark step_group inactive, then append result
                setMessages(prev => [
                  ...prev.map(m =>
                    m.type === 'step_group' && m.isActive ? { ...m, isActive: false } : m
                  ),
                  {
                    type: 'result',
                    content: data.message,
                    csv_file: data.csv_file,
                    row_count: data.row_count
                  }
                ])
                loadWorkspaces()
                setCurrentPlan(null)
              } else if (data.type === 'error') {
                setMessages(prev => [
                  ...prev.map(m =>
                    m.type === 'step_group' && m.isActive ? { ...m, isActive: false } : m
                  ),
                  { type: 'error', content: data.message }
                ])
              } else if (data.type === 'execution_stats') {
                if (data.total_cost_usd != null) {
                  // execution_stats now includes planning + execution costs
                  // (planning_stats is threaded into the Execution constructor), 
                  // so overwrite rather than accumulate.
                  setQueryCostUsd(data.total_cost_usd)
                }
                console.debug('Execution stats received:', data)
              }
            } catch (e) {
              console.error('Error parsing SSE data:', e)
            }
          }
        }
      }
    } catch (error) {
      setMessages(prev => prev.map(m =>
        m.type === 'step_group' && m.isActive ? { ...m, isActive: false } : m
      ))
      if (error && error.type === 'API_KEY_MISSING') {
        setMessages(prev => [...prev, { type: 'error', content: `${error.message} Please go to the Settings page to configure your keys.` }])
      } else if (error.name !== 'AbortError') {
        console.error('Error executing query:', error)
        setMessages(prev => [...prev, { type: 'error', content: 'Failed to execute query. Please try again.' }])
      }
    } finally {
      setIsLoading(false)
      abortControllerRef.current = null
    }
  }

  // ─── Notebook handlers ────────────────────────────────────────────

  const handleExecuteInJupyter = async (planToUse) => {
    const plan = planToUse || currentPlan
    if (!plan) return

    // Clear any lingering client-side error messages
    stripTrailingErrors()

    if (selectedDatasets.size === 0) {
      setMessages(prev => [...prev, { type: 'error', content: 'Please select at least one dataset before executing.' }])
      return
    }

    let currentSessionId = sessionId
    if (!currentWorkspaceId || !currentSessionId) {
      try {
        const token = await getValidToken()
        if (!token) return
        const datasetIds = Array.from(selectedDatasets).map(id => parseInt(id))
        const wsResp = await workspacesApi.create({ title: 'Untitled Workspace', dataset_ids: datasetIds.join(',') }, token)
        const ws = wsResp.data
        setCurrentWorkspaceId(ws.id)
        const firstConv = ws.conversations?.[0]
        if (firstConv) {
          setCurrentConversationId(firstConv.id)
          currentSessionId = firstConv.session_id
          setSessionId(currentSessionId)
        }
        loadWorkspaces()
      } catch (err) {
        console.error('Error auto-creating workspace:', err)
        return
      }
    }

    try {
      const token = await getValidToken()
      if (!token) return

      const datasetIds = Array.from(selectedDatasets).map(id => parseInt(id))
      const nextCount = notebookCountRef.current + 1
      const label = `Notebook ${nextCount}`
      // Prefer lastQuery; fall back to the last user message in the chat
      const queryText = lastQuery || messages.findLast(m => m.type === 'user')?.content || ''
      const response = await notebookApi.create({
        query: queryText,
        dataset_ids: datasetIds,
        session_id: currentSessionId,
        plan: plan,
        cost_budget: costBudget,
        workspace_id: currentWorkspaceId || undefined,
        conversation_id: currentConversationId || undefined,
        label,
      }, token)

      const { notebook_id, query, cells } = response.data
      notebookCountRef.current = nextCount

      // Store notebook state
      setNotebooks(prev => ({
        ...prev,
        [notebook_id]: { notebookId: notebook_id, query, cells }
      }))

      // Create a new tab and switch to it
      setTabs(prev => [
        ...prev,
        { id: notebook_id, label, type: 'notebook' }
      ])
      setActiveTabId(notebook_id)
      // Auto-dismiss plan visualizer to avoid visual overlap
      setShowPlanVisualizer(false)
    } catch (error) {
      console.error('Error creating notebook:', error)
      setMessages(prev => [...prev, {
        type: 'error',
        content: error?.response?.data?.detail || 'Failed to create notebook. Please try again.'
      }])
    }
  }

  const handleNotebookUpdate = (notebookId, updatedNotebook) => {
    setNotebooks(prev => ({ ...prev, [notebookId]: updatedNotebook }))
  }

  const handleCloseTab = async (tabId) => {
    // Can't close the chat tab
    if (tabId === 'chat') return

    // Close backend notebook state
    try {
      const token = await getValidToken()
      if (token) await notebookApi.close(tabId, token)
    } catch (error) {
      console.error('Error closing notebook:', error)
    }

    // Remove notebook + tab
    setNotebooks(prev => {
      const next = { ...prev }
      delete next[tabId]
      return next
    })
    setTabs(prev => prev.filter(t => t.id !== tabId))

    // If we closed the active tab, switch back to chat
    if (activeTabId === tabId) {
      setActiveTabId('chat')
      setShowPlanVisualizer(true)
    }
  }

  const handleTabSelect = (tabId) => {
    setActiveTabId(tabId)
    // Show plan visualizer only when on chat tab
    if (tabId === 'chat') {
      setShowPlanVisualizer(true)
    } else {
      setShowPlanVisualizer(false)
    }
  }

  // ─── Date formatting (sidebar) ────────────────────────────────────
  const formatDate = (dateString) => {
    const date = new Date(dateString)
    const now = new Date()
    const diffMs = now - date
    const diffMins = Math.floor(diffMs / 60000)
    const diffHours = Math.floor(diffMins / 60)
    const diffDays = Math.floor(diffHours / 24)
    if (diffMins < 1) return 'Just now'
    if (diffMins < 60) return `${diffMins}m ago`
    if (diffHours < 24) return `${diffHours}h ago`
    if (diffDays < 7) return `${diffDays}d ago`
    return date.toLocaleDateString()
  }

  // ─── Render ───────────────────────────────────────────────────────

  return (
    <div className="flex h-full w-full bg-white overflow-hidden">

      {/* 1. Left Sidebar */}
      <div className={`${sidebarCollapsed ? 'w-0 overflow-hidden' : 'w-80'} bg-gray-50 border-r border-gray-200 flex flex-col transition-all duration-300 flex-shrink-0`}>
        <div className="p-4">
          <button
            onClick={handleNewWorkspace}
            disabled={isLoading}
            className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-white hover:bg-gray-50 border border-gray-200 text-gray-700 rounded-xl transition-all shadow-sm font-medium"
          >
            <Plus className="w-5 h-5" />
            New Workspace
          </button>
        </div>

        {/* Datasets */}
        <div className="flex-1 flex flex-col min-h-0 px-4">
          <h2 className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-2">Datasets</h2>
          <div className="relative mb-3">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              value={datasetSearchQuery}
              onChange={(e) => setDatasetSearchQuery(e.target.value)}
              placeholder="Filter..."
              className="w-full pl-9 pr-3 py-1.5 bg-gray-200/50 border-none rounded-lg focus:ring-1 focus:ring-primary-500 text-sm"
            />
          </div>
          <div className="flex-1 overflow-y-auto space-y-1 pr-1">
            {filteredDatasets.map(dataset => (
              <div
                key={dataset.id}
                onClick={() => toggleDataset(dataset.id)}
                className={`flex items-start gap-3 p-2 rounded-lg cursor-pointer transition-colors ${
                  selectedDatasets.has(dataset.id) ? 'bg-primary-50 border border-primary-100' : 'hover:bg-gray-200'
                }`}
              >
                <div className="mt-0.5">
                  {selectedDatasets.has(dataset.id)
                    ? <CheckSquare className="w-4 h-4 text-primary-600" />
                    : <Square className="w-4 h-4 text-gray-400" />
                  }
                </div>
                <div className="min-w-0">
                  <p className="text-sm font-medium text-gray-800 truncate">{dataset.name}</p>
                  <p className="text-[11px] text-gray-500 line-clamp-2 leading-snug">{dataset.annotation}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="h-px bg-gray-200 mx-4 my-4" />

        {/* History */}
        <div className="flex-1 overflow-y-auto px-4 pb-4">
          <h2 className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-2">History</h2>
          <div className="space-y-1">
            {workspaces.map(ws => (
              <div
                key={ws.id}
                onClick={() => loadWorkspace(ws.id)}
                className={`group flex items-center justify-between p-2 rounded-lg cursor-pointer transition-colors ${
                  ws.id === currentWorkspaceId ? 'bg-gray-200' : 'hover:bg-gray-100'
                }`}
              >
                <div className="flex items-center gap-3 min-w-0">
                  <MessageSquare className="w-4 h-4 text-gray-400 flex-shrink-0" />
                  <span className="text-sm text-gray-600 truncate">{ws.title || 'Untitled'}</span>
                </div>
                <button
                  onClick={(e) => deleteWorkspace(ws.id, e)}
                  className="opacity-0 group-hover:opacity-100 p-1 hover:text-red-600 transition-opacity"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* 2. Main content area: TabBar + active view */}
      <div className="flex-1 flex flex-col min-w-0 relative">
        {/* Tab bar with inline sidebar toggle */}
        <div className="flex items-center">
          <button
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            className="flex-shrink-0 p-2 ml-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-all"
          >
            {sidebarCollapsed ? <ChevronRight className="w-5 h-5" /> : <ChevronLeft className="w-5 h-5" />}
          </button>
          <div className="flex-1 min-w-0">
            <TabBar
              tabs={tabs}
              activeId={activeTabId}
              onSelect={handleTabSelect}
              onClose={handleCloseTab}
            />
          </div>
        </div>

        {/* Tab content */}
        <div className="flex-1 flex min-h-0">
          {activeTabId === 'chat' ? (
            <ChatView
              messages={messages}
              isLoading={isLoading}
              isExecuting={isExecuting}
              sessionId={sessionId}
              inputQuery={inputQuery}
              setInputQuery={setInputQuery}
              costBudget={costBudget}
              setCostBudget={setCostBudget}
              queryCostUsd={queryCostUsd}
              onRequestPlan={handleRequestPlan}
              onExecutePlan={handleExecutePlan}
              onExecuteInJupyter={handleExecuteInJupyter}
              onCancel={handleCancel}
            />
          ) : notebooks[activeTabId] ? (
            <NotebookView
              notebook={notebooks[activeTabId]}
              onUpdate={handleNotebookUpdate}
            />
          ) : null}
        </div>
      </div>

      {/* 3. Plan Visualizer (right sidebar) — only shown on chat tab */}
      {currentPlan && activeTabId === 'chat' && (
        showPlanVisualizer ? (
          <div className="w-96 bg-white border-l border-gray-200 flex flex-col flex-shrink-0">
            <div className="p-4 border-b border-gray-200 flex items-center justify-between">
              <h2 className="text-xs font-bold text-gray-500 uppercase tracking-widest">Execution Plan</h2>
              <button onClick={() => setShowPlanVisualizer(false)} className="p-1 hover:bg-gray-100 rounded-lg">
                <X className="w-5 h-5 text-gray-400" />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto">
              <PlanVisualizer plan={currentPlan} />
            </div>
          </div>
        ) : (
          <button
            onClick={() => setShowPlanVisualizer(true)}
            title="Show execution plan"
            className="flex-shrink-0 border-l border-gray-200 bg-white hover:bg-gray-50 px-1.5 flex items-center justify-center transition-colors"
          >
            <ChevronLeft className="w-4 h-4 text-gray-400" />
          </button>
        )
      )}
    </div>
  )
}

export default UserWorkspacePage
