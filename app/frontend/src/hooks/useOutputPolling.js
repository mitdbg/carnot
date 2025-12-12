import { useState, useEffect, useRef } from 'react';

/**
 * Custom hook to poll for terminal output from the backend
 * @param {string} sessionId - The session ID to poll output for
 * @param {boolean} isActive - Whether to actively poll for updates
 * @returns {object} - Object containing output lines array
 */
export const useOutputPolling = (sessionId, isActive) => {
  const [outputLines, setOutputLines] = useState([]);
  const [lastLine, setLastLine] = useState(0);
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

    const pollOutput = async () => {
      try {
        const url = `/api/query/output/${sessionId}?last_line=${lastLine}`;
        
        const response = await fetch(url);
        const data = await response.json();

        if (data.lines && data.lines.length > 0) {
          setOutputLines(prev => [...prev, ...data.lines]);
          setLastLine(data.total_lines);
        }
      } catch (error) {
        console.error('Error polling output:', error);
      }
    };

    // Initial poll
    pollOutput();

    // Set up polling interval (every 500ms for more responsive output)
    intervalRef.current = setInterval(pollOutput, 500);

    // Cleanup
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [sessionId, isActive, lastLine]);

  // Reset when session changes
  useEffect(() => {
    setOutputLines([]);
    setLastLine(0);
  }, [sessionId]);

  return { outputLines };
};


