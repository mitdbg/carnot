import { useState, useEffect, useRef } from 'react';

/**
 * Custom hook to poll for progress updates from the backend
 * @param {string} sessionId - The session ID to poll progress for
 * @param {boolean} isActive - Whether to actively poll for updates
 * @returns {object} - Object containing progressEvents array
 */
export const useProgressPolling = (sessionId, isActive) => {
  const [progressEvents, setProgressEvents] = useState([]);
  const [lastTimestamp, setLastTimestamp] = useState(null);
  const intervalRef = useRef(null);

  useEffect(() => {
    if (!sessionId || !isActive) {
      // Clear interval if not active
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      return;
    }

    const pollProgress = async () => {
      try {
        const url = lastTimestamp
          ? `/api/query/progress/${sessionId}?since_timestamp=${encodeURIComponent(lastTimestamp)}`
          : `/api/query/progress/${sessionId}`;
        
        const response = await fetch(url);
        const data = await response.json();

        if (data.events && data.events.length > 0) {
          setProgressEvents(prev => [...prev, ...data.events]);
          // Update last timestamp to the newest event
          const newestTimestamp = data.events[data.events.length - 1].timestamp;
          setLastTimestamp(newestTimestamp);
        }
      } catch (error) {
        console.error('Error polling progress:', error);
      }
    };

    // Initial poll
    pollProgress();

    // Set up polling interval (every 1 second)
    intervalRef.current = setInterval(pollProgress, 1000);

    // Cleanup
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [sessionId, isActive, lastTimestamp]);

  return { progressEvents };
};


