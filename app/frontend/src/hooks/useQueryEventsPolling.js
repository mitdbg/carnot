import { useState, useEffect, useRef, useCallback } from 'react'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api'
const POLL_INTERVAL_MS = 1500

/**
 * Polls GET /api/query/events/{conversationId}?since_id=N every
 * POLL_INTERVAL_MS while shouldPoll is true.
 *
 * Returns accumulated events (each with id, event_type, source,
 * payload, step_cost_usd) and an isComplete flag.
 *
 * The caller (UserWorkspacePage) dispatches events into its existing
 * setMessages / setQueryCostUsd state using the same logic as the
 * SSE handlers.
 *
 * @param {number|null} conversationId  - DB id of the conversation to poll
 * @param {boolean}     shouldPoll      - Gate: only poll when true
 * @param {Function}    getValidToken   - Auth0 token getter from useApiToken()
 */
export function useQueryEventsPolling(conversationId, shouldPoll, getValidToken) {
  const [events, setEvents] = useState([])
  const [isComplete, setIsComplete] = useState(false)
  const lastIdRef = useRef(0)
  const intervalRef = useRef(null)

  const reset = useCallback(() => {
    setEvents([])
    setIsComplete(false)
    lastIdRef.current = 0
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
  }, [])

  useEffect(() => {
    if (!conversationId || !shouldPoll) {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
        intervalRef.current = null
      }
      return
    }

    const poll = async () => {
      try {
        const token = await getValidToken()
        if (!token) return

        const url = `${API_BASE_URL}/query/events/${conversationId}?since_id=${lastIdRef.current}`
        const resp = await fetch(url, {
          headers: { Authorization: `Bearer ${token}` },
        })
        if (!resp.ok) return

        const data = await resp.json()
        if (data.events && data.events.length > 0) {
          setEvents(prev => [...prev, ...data.events])
          lastIdRef.current = data.events[data.events.length - 1].id
        }
        if (data.is_complete) {
          setIsComplete(true)
          if (intervalRef.current) {
            clearInterval(intervalRef.current)
            intervalRef.current = null
          }
        }
      } catch (err) {
        console.error('[useQueryEventsPolling] poll error:', err)
      }
    }

    // Immediate catch-up fetch, then regular interval
    poll()
    intervalRef.current = setInterval(poll, POLL_INTERVAL_MS)

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
        intervalRef.current = null
      }
    }
  }, [conversationId, shouldPoll, getValidToken])

  return { events, isComplete, reset }
}
