import { useState, useEffect, useRef } from 'react'
import { Send, Database, CheckSquare, Square, AlertCircle, Loader2, XCircle, RotateCcw, MessageSquare, Trash2, ChevronLeft, ChevronRight, Search, Play, PenTool, X } from 'lucide-react'
import { useApiToken } from '../hooks/useApiToken'
import axios from 'axios'
import ProgressDisplay from '../components/ProgressDisplay'
import { conversationsApi, datasetsApi } from '../services/api'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api"

function PlanVisualizer({ plan }) {
  // Recursively flattens the tree into chronological steps
  const flattenPlan = (node, acc = new Map()) => {
    if (!node) return acc;

    // Recurse through parents first
    if (node.parents && node.parents.length > 0) {
      node.parents.forEach(parent => flattenPlan(parent, acc));
    }

    // Use a Map keyed by the serialized node to avoid duplicate steps 
    // (common in complex join branches)
    const nodeKey = JSON.stringify({ name: node.name, params: node.params });
    if (!acc.has(nodeKey)) {
      acc.set(nodeKey, {
        name: node.name,
        operator: node.params?.operator || "Source",
        description: node.params?.description || `Load dataset: ${node.name}`,
        parents: node.parents?.map(p => p.name) || [],
        details: node.params || {}
      });
    }

    return acc;
  };

  const stepsMap = plan ? flattenPlan(plan) : new Map();
  const steps = Array.from(stepsMap.values());

  if (steps.length === 0) return (
    <div className="flex flex-col items-center justify-center h-full text-gray-400 p-4 text-center">
      <p>No active execution plan.</p>
    </div>
  );

  return (
    <div className="p-4 overflow-y-auto h-full">
      <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-6">Logical DAG</h3>
      <div className="space-y-0">
        {steps.map((step, idx) => {
          const isCombination = step.parents.length > 1;
          
          return (
            <div key={idx} className="flex gap-4">
              {/* Visual Connector Line */}
              <div className="flex flex-col items-center">
                <div className={`w-3 h-3 rounded-full border-2 transition-colors ${
                  idx === steps.length - 1 
                    ? 'bg-primary-500 border-primary-500' 
                    : 'bg-white border-gray-300'
                }`} />
                {idx !== steps.length - 1 && <div className="w-0.5 h-full bg-gray-200" />}
              </div>

              {/* Step Content */}
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

                  {/* Multi-parent branch indicator */}
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

                  {/* Parameter Details */}
                  {step.details.condition && (
                    <div className="mt-2 text-[11px] bg-gray-50 p-1.5 rounded border border-gray-100 font-mono text-gray-600 break-all">
                      <span className="text-blue-600 font-bold">WHERE:</span> {step.details.condition}
                    </div>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function UserChatPage() {
  const getValidToken = useApiToken();
  const [datasets, setDatasets] = useState([])
  const [selectedDatasets, setSelectedDatasets] = useState(new Set())
  const [messages, setMessages] = useState([])
  const [inputQuery, setInputQuery] = useState('')
  const [isExecuting, setIsExecuting] = useState(false);
  const [isLoading, setIsLoading] = useState(false)
  const [sessionId, setSessionId] = useState(null)
  const [conversations, setConversations] = useState([])
  const [currentConversationId, setCurrentConversationId] = useState(null)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [datasetSearchQuery, setDatasetSearchQuery] = useState('')
  const [currentPlan, setCurrentPlan] = useState(null);
  const [lastQuery, setLastQuery] = useState('');
  const messagesEndRef = useRef(null)
  const abortControllerRef = useRef(null)

  // Generate session ID on component mount
  useEffect(() => {
    generateNewSession()
    loadConversations()
  }, [])

  useEffect(() => {
    loadDatasets()
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const loadConversations = async () => {
    try {
      const token = await getValidToken();
      if (!token) return;

      const response = await conversationsApi.list(token)
      setConversations(response.data)
    } catch (error) {
      console.error('Error loading conversations:', error)
    }
  }

  const loadConversation = async (conversationId) => {
    try {
      const token = await getValidToken();
      if (!token) return;

      const response = await conversationsApi.get(conversationId, token)
      const conversation = response.data
      
      // Set session ID and messages
      setSessionId(conversation.session_id)
      setCurrentConversationId(conversation.id)
      
      // Convert database messages to frontend format
      const formattedMessages = conversation.messages.map(msg => ({
        type: msg.role,
        content: msg.content,
        csv_file: msg.csv_file,
        row_count: msg.row_count
      }))
      setMessages(formattedMessages)
      
      if (conversation.dataset_ids) {
        const datasetIds = conversation.dataset_ids.split(',').map(id => parseInt(id))
        setSelectedDatasets(new Set(datasetIds))
      }
    } catch (error) {
      console.error('Error loading conversation:', error)
    }
  }

  const deleteConversation = async (conversationId, e) => {
    e.stopPropagation() // Prevent loading the conversation
    
    if (!window.confirm('Are you sure you want to delete this conversation?')) {
      return
    }
    
    try {
      const token = await getValidToken();
      if (!token) return;

      await conversationsApi.delete(conversationId, token)
      
      // If we deleted the current conversation, create a new session
      if (conversationId === currentConversationId) {
        generateNewSession()
      }
      
      // Reload conversations list
      loadConversations()
    } catch (error) {
      console.error('Error deleting conversation:', error)
    }
  }
  
  const generateNewSession = () => {
    const newSessionId = crypto.randomUUID()
    setSessionId(newSessionId)
    setMessages([])
    setCurrentConversationId(null)
    setSelectedDatasets(new Set())
  }

  const loadDatasets = async () => {
    try {
      const token = await getValidToken();
      if (!token) return;

      const response = await datasetsApi.list(token)
      setDatasets(response.data)
    } catch (error) {
      console.error('Error loading datasets:', error)
    }
  }

  const toggleDataset = (datasetId) => {
    setSelectedDatasets(prev => {
      const newSet = new Set(prev)
      if (newSet.has(datasetId)) {
        newSet.delete(datasetId)
      } else {
        newSet.add(datasetId)
      }
      return newSet
    })
  }

  // Filter datasets based on search query
  const filteredDatasets = datasets.filter(dataset => {
    if (!datasetSearchQuery.trim()) return true
    const query = datasetSearchQuery.toLowerCase()
    return (
      dataset.name.toLowerCase().includes(query) ||
      dataset.annotation.toLowerCase().includes(query)
    )
  })

  const handleCancel = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      setMessages(prev => [...prev, {
        type: 'error',
        content: 'Query cancelled by user.'
      }])
      setIsLoading(false)
      abortControllerRef.current = null
    }
  }

  const handleRequestPlan = async (e) => {
    if (e) e.preventDefault();
    if (!inputQuery.trim()) {
      return
    }

    if (selectedDatasets.size === 0) {
      setMessages(prev => [...prev, {
        type: 'error',
        content: 'Please select at least one dataset before submitting a query.'
      }])
      return
    }

    let currentSessionId = sessionId || crypto.randomUUID();
    if (!sessionId) setSessionId(currentSessionId);

    const queryToPlan = inputQuery;
    setLastQuery(queryToPlan);

    const userMsg = { type: 'user', content: queryToPlan };
    setMessages(prev => [...prev, userMsg]);
    setInputQuery('');
    setIsLoading(true);
    setIsExecuting(false);
    try {
      const token = await getValidToken();
      if (!token) return;
      const datasetIds = Array.from(selectedDatasets).map(id => parseInt(id));
      const response = await axios.post(`${API_BASE_URL}/query/plan`, {
        query: queryToPlan,
        dataset_ids: datasetIds,
        session_id: currentSessionId,
        plan: currentPlan
      }, {
        headers: { Authorization: `Bearer ${token}` }
      });

      setMessages(prev => [...prev, {
        type: 'assistant',
        content: response.data.natural_language_plan,
        isPlanConfirmation: true, // Flag to show "Execute" buttons
        attachedPlan: response.data.plan
      }]);
      setCurrentPlan(response.data.plan); // TODO: maybe redundant now that we use attachedPlan?
    } catch (error) {
      const errorData = error.response?.data?.detail;
      if (errorData?.type === 'API_KEY_MISSING') {
        setMessages(prev => [...prev, {
          type: 'error',
          content: `${errorData.message} Please go to the Settings page to configure your keys.`
        }])
      } else if (error.name !== 'CanceledError') {
        console.error('Error planning query:', error)
        setMessages(prev => [...prev, {
          type: 'error',
          content: errorData?.message || 'Failed to generate plan. Please try again.'
        }])
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleExecutePlan = async (planToUse) => {
    const plan = planToUse || currentPlan;
    setIsLoading(true);
    setIsExecuting(true);

    if (!lastQuery) {
        setIsLoading(false);
        return;
    }

    if (selectedDatasets.size === 0) {
      setMessages(prev => [...prev, {
        type: 'error',
        content: 'Please select at least one dataset before submitting a query.'
      }])
      setIsLoading(false);
      return
    }

    // if sessionId is null (due to initial state/race condition), generate a new one immediately.
    let currentSessionId = sessionId;
    if (!currentSessionId) {
      currentSessionId = crypto.randomUUID();
      setSessionId(currentSessionId); 
    }

    try {
      // fetch access token
      const token = await getValidToken();
      if (!token) return;

      // Create abort controller for this request
      abortControllerRef.current = new AbortController()
      const datasetIds = Array.from(selectedDatasets).map(id => parseInt(id));
      const response = await fetch(`${API_BASE_URL}/query/execute`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify({
          query: lastQuery,
          dataset_ids: datasetIds,
          session_id: currentSessionId,
          plan: plan
        }),
        signal: abortControllerRef.current.signal
      })

      // handle non-streaming errors (e.g., 400 No API Keys)
      if (!response.ok) {
        let errorData;
        try {
          // response body is the FastAPI JSON error structure
          errorData = await response.json();
        } catch (e) {
          throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
        }
        
        // Pass the detail object so we can inspect its 'type' in the catch block
        // If errorData.detail is the structured dictionary, we throw it. Otherwise, fall back to the generic error string.
        throw errorData.detail || new Error(errorData.message || 'An unknown server error occurred.');
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
            try {
              const data = JSON.parse(line.slice(6))
              
              // Update session_id if received from server
              if (data.session_id && data.session_id !== sessionId) {
                setSessionId(data.session_id)
              }
              
              if (data.type === 'status') {
                setMessages(prev => [...prev, {
                  type: 'status',
                  content: data.message
                }])
              } else if (data.type === 'result') {
                setMessages(prev => [...prev, {
                  type: 'result',
                  content: data.message,
                  csv_file: data.csv_file,
                  row_count: data.row_count
                }])
                // Reload conversations after query completes
                loadConversations()
                setCurrentPlan(null);
              } else if (data.type === 'error') {
                setMessages(prev => [...prev, {
                  type: 'error',
                  content: data.message
                }])
              }
            } catch (e) {
              console.error('Error parsing SSE data:', e)
            }
          }
        }
      }
    } catch (error) {
      if (error && error.type === 'API_KEY_MISSING') {
        console.warn('API Key missing:', error.message)
        setMessages(prev => [...prev, {
          type: 'error',
          // Use the specific message from the server response
          content: `${error.message} Please go to the Settings page to configure your keys.`
        }])
      }
      else if (error.name !== 'AbortError') {
        console.error('Error executing query:', error)
        setMessages(prev => [...prev, {
          type: 'error',
          content: 'Failed to execute query. Please try again.'
        }])
      }
    } finally {
      setIsLoading(false)
      abortControllerRef.current = null
    }
  }

  const renderMessage = (message, index) => {
    const isLastMessage = index === messages.length - 1;
    switch (message.type) {
      case 'user':
        return (
          <div key={index} className="flex justify-end mb-4">
            <div className="bg-primary-500 text-white rounded-lg px-4 py-2 max-w-[70%]">
              {message.content}
            </div>
          </div>
        )
      
      case 'assistant':
        return (
          <div key={index} className="flex flex-col items-start mb-4">
            <div className="bg-white border border-gray-200 p-4 rounded-lg max-w-[85%] shadow-sm">
              <div className="prose prose-sm text-gray-700 whitespace-pre-wrap">
                {message.content}
              </div>
              
              {message.isPlanConfirmation && isLastMessage && !isLoading && (
                <div className="mt-4 flex gap-3 border-t pt-4">
                  <button
                    onClick={handleExecutePlan(message.attachedPlan)}
                    className="flex items-center gap-2 bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-md text-sm font-medium transition-colors"
                  >
                    <Play className="w-4 h-4" /> Execute Plan
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
      
      default:
        return null
    }
  }

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

  return (
    <div className="h-[calc(100vh-4rem)] flex overflow-hidden bg-white">
      {/* 1. Left sidebar - Conversation history */}
      <div className={`${sidebarCollapsed ? 'w-16' : 'w-64'} bg-gray-50 border-r border-gray-200 flex flex-col transition-all duration-300`}>
        <div className="p-4 border-b border-gray-200 flex items-center justify-between">
          {!sidebarCollapsed && <h2 className="text-lg font-bold text-gray-800">Conversations</h2>}
          <button
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            className="p-1 hover:bg-gray-200 rounded transition-colors"
            title={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {sidebarCollapsed ? <ChevronRight className="w-5 h-5" /> : <ChevronLeft className="w-5 h-5" />}
          </button>
        </div>
        
        {!sidebarCollapsed && (
          <div className="flex-1 overflow-y-auto p-2">
            {conversations.length === 0 ? (
              <p className="text-sm text-gray-500 text-center mt-4">No conversations yet</p>
            ) : (
              <div className="space-y-2">
                {conversations.map(conv => (
                  <div
                    key={conv.id}
                    onClick={() => loadConversation(conv.id)}
                    className={`p-3 rounded-lg cursor-pointer transition-colors group hover:bg-gray-200 ${
                      conv.id === currentConversationId ? 'bg-primary-100 border border-primary-300' : 'bg-white border border-gray-200'
                    }`}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <MessageSquare className="w-4 h-4 text-gray-500 flex-shrink-0" />
                          <p className="text-sm font-medium text-gray-800 truncate">
                            {conv.title || 'Untitled'}
                          </p>
                        </div>
                        <p className="text-xs text-gray-500">{formatDate(conv.updated_at)}</p>
                        {conv.message_count > 0 && (
                          <p className="text-xs text-gray-400 mt-1">{conv.message_count} messages</p>
                        )}
                      </div>
                      <button
                        onClick={(e) => deleteConversation(conv.id, e)}
                        className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-100 rounded transition-all"
                        title="Delete conversation"
                      >
                        <Trash2 className="w-4 h-4 text-red-500" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* 2. Middle-Left: Chat (Width Adjust) */}
      <div className="flex-1 flex flex-col min-w-0 border-r border-gray-200">
        {/* Chat header */}
        <div className="bg-white border-b border-gray-200 px-6 py-4">
          <div>
            <h1 className="text-2xl font-bold text-gray-800">Query Chat</h1>
            <p className="text-sm text-gray-600 mt-1">
              Ask questions about your data {sessionId && <span className="text-xs text-gray-400">(Session: {sessionId.slice(0, 8)}...)</span>}
            </p>
          </div>
          <button
            onClick={generateNewSession}
            disabled={isLoading}
            className="flex items-center gap-2 px-4 py-2 bg-gray-100 hover:bg-gray-200 disabled:bg-gray-50 disabled:cursor-not-allowed text-gray-700 rounded-lg transition-colors"
            title="Start a new conversation"
          >
            <RotateCcw className="w-4 h-4" />
            New Conversation
          </button>
        </div>

        {/* Messages container */}
        <div className="flex-1 overflow-y-auto px-6 py-4 bg-gray-50">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <Database className="w-16 h-16 text-gray-300 mb-4" />
              <h2 className="text-xl font-semibold text-gray-600 mb-2">
                No messages yet
              </h2>
              <p className="text-gray-500 max-w-md">
                Select datasets from the right panel and start asking questions about your data
              </p>
            </div>
          )}
          
          {messages.map((message, index) => renderMessage(message, index))}
          
          {/* Show progress display when loading - after messages */}
          {isLoading && isExecuting && sessionId && (
            <ProgressDisplay sessionId={sessionId} isActive={isLoading} />
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Input area */}
        <div className="bg-white border-t border-gray-200 px-6 py-4">
          <form onSubmit={handleRequestPlan} className="flex gap-2">
            <input
              type="text"
              value={inputQuery}
              onChange={(e) => setInputQuery(e.target.value)}
              placeholder={currentPlan ? "Refine the query plan..." : "Type your query here..."}
              className="flex-1 border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500"
              disabled={isLoading}
            />
            <button
              type="submit"
              className="bg-primary-500 hover:bg-primary-600 text-white px-4 py-2 rounded-lg flex items-center gap-2 disabled:bg-primary-300 disabled:cursor-not-allowed transition-colors"
              disabled={isLoading}
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  <Send className="w-4 h-4" />
                  Send
                </>
              )}
            </button>
            {isLoading && (
              <button
                type="button"
                onClick={handleCancel}
                className="bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded-lg flex items-center gap-2 transition-colors"
              >
                <XCircle className="w-4 h-4" />
                Cancel
              </button>
            )}
          </form>
        </div>
      </div>

      {/* 3. NEW: Middle-Right: DAG Visualizer */}
      {currentPlan && (
        <div className="w-96 bg-white border-r border-gray-200 flex flex-col min-w-0">
          <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between">
            <h2 className="text-lg font-bold text-gray-800">Execution Plan</h2>
            <button 
              onClick={() => setCurrentPlan(null)}
              className="p-1 hover:bg-gray-100 rounded-full transition-colors"
            >
              <X className="w-5 h-5 text-gray-500" />
            </button>
          </div>
          <div className="flex-1 overflow-y-auto">
            <PlanVisualizer plan={currentPlan} />
          </div>
        </div>
      )}

      {/* 4. Right side - Dataset selection */}
      <div className="w-80 bg-white flex flex-col">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-xl font-bold text-gray-800">Datasets</h2>
          <p className="text-sm text-gray-600 mt-1 mb-3">
            Select datasets to query
          </p>
          {/* Search bar */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              value={datasetSearchQuery}
              onChange={(e) => setDatasetSearchQuery(e.target.value)}
              placeholder="Search datasets..."
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 text-sm"
            />
          </div>
        </div>
        <div className="flex-1 overflow-y-auto p-6">
          {datasets.length === 0 ? (
            <p className="text-gray-500">No datasets available.</p>
          ) : filteredDatasets.length === 0 ? (
            <p className="text-gray-500">No datasets match your search.</p>
          ) : (
            <ul className="space-y-3">
              {filteredDatasets.map(dataset => (
                <li
                  key={dataset.id}
                  className="flex items-start justify-between p-3 bg-gray-50 rounded-lg border border-gray-200"
                >
                  <div className="flex-1 min-w-0">
                    <p className="font-semibold text-gray-800 truncate" title={dataset.name}>
                      {dataset.name}
                    </p>
                    {/* The line-clamp-3 class limits text to exactly 3 lines before adding '...' */}
                    <p 
                      className="text-sm text-gray-600 mt-1 line-clamp-3 leading-relaxed" 
                      title={dataset.annotation}
                    >
                      {dataset.annotation}
                    </p>
                  </div>
                  <button
                    onClick={() => toggleDataset(dataset.id)}
                    className="p-2 ml-2 rounded-full text-gray-500 hover:bg-gray-200 transition-colors flex-shrink-0"
                  >
                    {selectedDatasets.has(dataset.id) ? (
                      <CheckSquare className="w-5 h-5 text-primary-500" />
                    ) : (
                      <Square className="w-5 h-5" />
                    )}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  )
}

export default UserChatPage
