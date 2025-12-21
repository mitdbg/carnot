import { useState, useEffect, useRef } from 'react'
import { Send, Database, CheckSquare, Square, AlertCircle, Loader2, XCircle, RotateCcw, MessageSquare, Trash2, ChevronLeft, ChevronRight, Search } from 'lucide-react'
import { useApiToken } from '../hooks/useApiToken';
import axios from 'axios'
import ProgressDisplay from '../components/ProgressDisplay'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api"

function UserChatPage() {
  const getValidToken = useApiToken();
  const [datasets, setDatasets] = useState([])
  const [selectedDatasets, setSelectedDatasets] = useState(new Set())
  const [messages, setMessages] = useState([])
  const [inputQuery, setInputQuery] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [sessionId, setSessionId] = useState(null)
  const [conversations, setConversations] = useState([])
  const [currentConversationId, setCurrentConversationId] = useState(null)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [datasetSearchQuery, setDatasetSearchQuery] = useState('')
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
      // get the raw JWT Access Token
      const token = await getValidToken();

      const response = await axios.get(`${API_BASE_URL}/conversations/`, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });
      setConversations(response.data)
    } catch (error) {
      console.error('Error loading conversations:', error)
    }
  }

  const loadConversation = async (conversationId) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/conversations/${conversationId}`)
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
      await axios.delete(`${API_BASE_URL}/conversations/${conversationId}`)
      
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
      const response = await axios.get(`${API_BASE_URL}/datasets/`)
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

  const handleSubmit = async (e) => {
    e.preventDefault()
    
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

    // if sessionId is null (due to initial state/race condition), generate a new one immediately.
    let currentSessionId = sessionId;
    if (!currentSessionId) {
      const newSessionId = crypto.randomUUID();
      setSessionId(newSessionId); 
      currentSessionId = newSessionId;
    }

    // Add user message
    const userMessage = {
      type: 'user',
      content: inputQuery
    }
    setMessages(prev => [...prev, userMessage])
    setInputQuery('')
    setIsLoading(true)

    try {
      // fetch access token
      const token = await getValidToken();

      // Create abort controller for this request
      abortControllerRef.current = new AbortController()

      const response = await fetch(`${API_BASE_URL}/query/execute`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify({
          query: userMessage.content,
          dataset_ids: Array.from(selectedDatasets),
          session_id: sessionId
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
        console.log('Error Name:', error.name)
        console.log('Error Message:', error.message)
        console.log('Error Keys:', Object.keys(error))
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
          <div key={index} className="flex justify-start mb-4">
            <div className="bg-gray-100 p-3 rounded-lg max-w-[70%] whitespace-pre-wrap">
              {message.content}
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
    <div className="h-[calc(100vh-4rem)] flex">
      {/* Left sidebar - Conversation history */}
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

      {/* Middle - Chat */}
      <div className="flex-1 flex flex-col border-r border-gray-200">
        {/* Chat header */}
        <div className="bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between">
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
          {isLoading && sessionId && (
            <ProgressDisplay sessionId={sessionId} isActive={isLoading} />
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Input area */}
        <div className="bg-white border-t border-gray-200 px-6 py-4">
          <form onSubmit={handleSubmit} className="flex gap-2">
            <input
              type="text"
              value={inputQuery}
              onChange={(e) => setInputQuery(e.target.value)}
              placeholder="Type your query here..."
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

      {/* Right side - Dataset selection */}
      <div className="w-1/3 bg-white border-l border-gray-200 flex flex-col">
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
                  className="flex items-center justify-between p-3 bg-gray-50 rounded-lg border border-gray-200"
                >
                  <div className="flex-1">
                    <p className="font-semibold text-gray-800">{dataset.name}</p>
                    <p className="text-sm text-gray-600">{dataset.annotation}</p>
                  </div>
                  <button
                    onClick={() => toggleDataset(dataset.id)}
                    className="p-2 rounded-full text-gray-500 hover:bg-gray-200 transition-colors"
                    title={selectedDatasets.has(dataset.id) ? 'Deselect dataset' : 'Select dataset'}
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
