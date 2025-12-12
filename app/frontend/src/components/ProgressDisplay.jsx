import React, { useRef, useEffect } from 'react';
import { useOutputPolling } from '../hooks/useOutputPolling';

/**
 * Component to display real-time terminal output for query execution
 */
const ProgressDisplay = ({ sessionId, isActive }) => {
  const { outputLines } = useOutputPolling(sessionId, isActive);
  const outputEndRef = useRef(null);

  useEffect(() => {
    outputEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [outputLines]);

  if (outputLines.length === 0) {
    return null;
  }

  return (
    <div className="bg-gray-900 rounded-lg shadow-md p-4 mb-6 max-h-96 overflow-y-auto">
      <h3 className="text-lg font-semibold mb-3 text-gray-200 sticky top-0 bg-gray-900 pb-2">
        Execution Output
      </h3>
      <pre className="text-sm text-gray-200 font-mono whitespace-pre-wrap break-words">
        {outputLines.map((line, index) => (
          <div key={index}>{line}</div>
        ))}
        <div ref={outputEndRef} />
      </pre>
    </div>
  );
};

export default ProgressDisplay;
