import { useState } from 'react'
import { useAuth0 } from '@auth0/auth0-react';
import { Send, Loader2, Bot, User, CheckCircle } from 'lucide-react'
import { searchApi } from '../../services/api'

function SearchChatbot({ onSelectFiles }) {
  const { getAccessTokenSilently } = useAuth0();
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Hi! I can help you find files. Try asking me something like "Find all email files" or "Search for files about investments".',
    },
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [lastResults, setLastResults] = useState([])

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!input.trim() || loading) return

    const userMessage = input.trim()
    setInput('')

    // Add user message
    setMessages((prev) => [...prev, { role: 'user', content: userMessage }])
    setLoading(true)

    try {
      // fetch access token
      const token = await getAccessTokenSilently();

      // search for files
      const response = await searchApi.search(userMessage, null, token);
      const results = response.data
      setLastResults(results)

      // add assistant response
      if (results.length > 0) {
        setMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: `I found ${results.length} file(s) matching your query. Click "Add to Selection" to include them in your dataset.`,
            results: results,
          },
        ])
      } else {
        setMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: "I couldn't find any files matching your query. Try rephrasing or using different keywords.",
          },
        ])
      }
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: 'Sorry, I encountered an error searching for files: ' + err.message,
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  const handleAddToSelection = (results) => {
    onSelectFiles(results)
    setMessages((prev) => [
      ...prev,
      {
        role: 'assistant',
        content: `Added ${results.length} file(s) to your selection!`,
      },
    ])
  }

  return (
    <div className="bg-white rounded-lg shadow-md flex flex-col h-full min-h-[500px]">
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <h2 className="text-xl font-semibold text-gray-800 flex items-center gap-2">
          <Bot className="w-5 h-5 text-primary-500" />
          AI File Search
        </h2>
        <p className="text-sm text-gray-600 mt-1">Ask me to find specific files</p>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex gap-3 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            {message.role === 'assistant' && (
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary-100 flex items-center justify-center">
                <Bot className="w-5 h-5 text-primary-600" />
              </div>
            )}

            <div
              className={`max-w-[80%] rounded-lg px-4 py-2 ${
                message.role === 'user'
                  ? 'bg-primary-600 text-white'
                  : 'bg-gray-100 text-gray-800'
              }`}
            >
              <p className="text-sm whitespace-pre-wrap">{message.content}</p>

              {/* Display search results */}
              {message.results && message.results.length > 0 && (
                <div className="mt-3 space-y-2">
                  <div className="max-h-48 overflow-y-auto space-y-2">
                    {message.results.slice(0, 5).map((result, idx) => (
                      <div
                        key={idx}
                        className="bg-white border border-gray-200 rounded p-2 text-xs"
                      >
                        <div className="font-medium text-gray-800">{result.file_name}</div>
                        <div className="text-gray-500 text-xs truncate">{result.file_path}</div>
                        {result.snippet && (
                          <div className="text-gray-600 mt-1 text-xs line-clamp-2">
                            {result.snippet}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>

                  <button
                    onClick={() => handleAddToSelection(message.results)}
                    className="w-full mt-2 flex items-center justify-center gap-2 px-3 py-2 bg-primary-600 text-white text-sm rounded hover:bg-primary-700 transition-colors"
                  >
                    <CheckCircle className="w-4 h-4" />
                    Add {message.results.length} file(s) to Selection
                  </button>
                </div>
              )}
            </div>

            {message.role === 'user' && (
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center">
                <User className="w-5 h-5 text-gray-600" />
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="flex gap-3">
            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary-100 flex items-center justify-center">
              <Bot className="w-5 h-5 text-primary-600" />
            </div>
            <div className="bg-gray-100 rounded-lg px-4 py-2">
              <Loader2 className="w-5 h-5 animate-spin text-gray-600" />
            </div>
          </div>
        )}
      </div>

      {/* Input */}
      <div className="p-4 border-t border-gray-200">
        <form onSubmit={handleSubmit} className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask me to find files..."
            disabled={loading}
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent disabled:bg-gray-100"
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Send className="w-5 h-5" />
          </button>
        </form>
      </div>
    </div>
  )
}

export default SearchChatbot

