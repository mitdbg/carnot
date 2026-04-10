import { useState } from 'react'
import { Play, CheckCircle, AlertCircle, Loader2, Clock, Trash2, ChevronUp, ChevronDown, RotateCcw } from 'lucide-react'
import CodeEditor from './CodeEditor'

/**
 * Renders a single notebook cell: header with operator name/status,
 * an editable syntax-highlighted code block, a run button, and an output area.
 *
 * Props:
 *   cell       – cell descriptor dict from the backend
 *   index      – 0-based cell index (for display as "Cell N")
 *   onRun      – (cellId, code) => void
 *   onDelete   – (cellId) => void
 *   onMove     – (cellId, direction: 'up' | 'down') => void
 *   isRunning  – whether this cell is currently executing
 *   disabled   – whether the run button should be disabled
 *   isFirst    – whether this is the first cell (cannot move up)
 *   isLast     – whether this is the last cell (cannot move down)
 *   canDelete  – whether the cell can be deleted
 */
function NotebookCell({ cell, index, onRun, onDelete, onMove, isRunning, disabled, isFirst, isLast, canDelete }) {
  const [editedCode, setEditedCode] = useState(null) // null = not edited

  const statusConfig = {
    pending:  { icon: Clock,       color: 'text-gray-400',  bg: 'bg-gray-50',   label: 'Pending' },
    running:  { icon: Loader2,     color: 'text-blue-500',  bg: 'bg-blue-50',   label: 'Running' },
    success:  { icon: CheckCircle, color: 'text-green-500', bg: 'bg-green-50',  label: 'Success' },
    error:    { icon: AlertCircle, color: 'text-red-500',   bg: 'bg-red-50',    label: 'Error' },
  }

  const status = statusConfig[cell.status] || statusConfig.pending
  const StatusIcon = status.icon

  const currentCode = editedCode !== null ? editedCode : cell.code
  const isModified = editedCode !== null && editedCode !== cell.original_code

  const handleRun = () => {
    onRun(cell.cell_id, isModified ? editedCode : null)
  }

  const handleReset = () => {
    setEditedCode(null)
  }

  return (
    <div className="border border-gray-200 rounded-lg overflow-hidden bg-white shadow-sm">
      {/* Cell Header */}
      <div className="flex items-center justify-between px-3 py-2 bg-gray-50 border-b border-gray-200">
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono text-gray-400">
            [{index + 1}]
          </span>
          <span className={`text-[10px] font-bold px-2 py-0.5 rounded uppercase font-mono ${
            cell.cell_type === 'dataset'
              ? 'bg-blue-100 text-blue-700'
              : cell.cell_type === 'reasoning'
                ? 'bg-purple-100 text-purple-700'
                : 'bg-primary-50 text-primary-700'
          }`}>
            {cell.operator_type || cell.cell_type}
          </span>
          <span className="text-sm font-medium text-gray-700 truncate max-w-[300px]">
            {cell.operator_name}
          </span>
          {isModified && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-yellow-100 text-yellow-700 font-medium">
              modified
            </span>
          )}
        </div>
        <div className="flex items-center gap-1">
          {/* Move up/down */}
          <button
            onClick={() => onMove(cell.cell_id, 'up')}
            disabled={isFirst || disabled || cell.cell_type === 'dataset'}
            className="p-1 text-gray-400 hover:text-gray-600 hover:bg-gray-200 rounded
              disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            title="Move up"
          >
            <ChevronUp className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => onMove(cell.cell_id, 'down')}
            disabled={isLast || disabled || cell.cell_type === 'dataset' || cell.cell_type === 'reasoning'}
            className="p-1 text-gray-400 hover:text-gray-600 hover:bg-gray-200 rounded
              disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            title="Move down"
          >
            <ChevronDown className="w-3.5 h-3.5" />
          </button>

          {/* Reset code */}
          {isModified && (
            <button
              onClick={handleReset}
              className="p-1 text-yellow-500 hover:text-yellow-700 hover:bg-yellow-50 rounded transition-colors"
              title="Reset to original"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>
          )}

          {/* Status */}
          <div className={`flex items-center gap-1 text-[11px] mx-1 ${status.color}`}>
            <StatusIcon className={`w-3.5 h-3.5 ${cell.status === 'running' ? 'animate-spin' : ''}`} />
            {status.label}
          </div>

          {/* Run button */}
          <button
            onClick={handleRun}
            disabled={disabled || isRunning}
            className="flex items-center gap-1 px-2 py-1 text-[11px] font-medium rounded
              bg-green-600 hover:bg-green-700 text-white
              disabled:bg-gray-300 disabled:cursor-not-allowed
              transition-colors"
          >
            {isRunning
              ? <Loader2 className="w-3 h-3 animate-spin" />
              : <Play className="w-3 h-3" />
            }
            Run
          </button>

          {/* Delete button */}
          {canDelete && (
            <button
              onClick={() => onDelete(cell.cell_id)}
              disabled={disabled}
              className="p-1 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded
                disabled:opacity-30 disabled:cursor-not-allowed transition-colors ml-1"
              title="Delete cell"
            >
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      {/* Code Block — editable with syntax highlighting */}
      <CodeEditor
        code={currentCode}
        onChange={setEditedCode}
        readOnly={isRunning}
        onRun={(!disabled && !isRunning) ? handleRun : undefined}
        error={cell.status === 'error' ? cell.error : null}
      />

      {/* Output Area */}
      {cell.status === 'success' && cell.output && (
        <div className="border-t border-gray-200 p-3 bg-green-50/30">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-[10px] font-bold text-gray-500 uppercase">Output</span>
            <span className="text-[11px] text-gray-500">
              {cell.output.items_count} item{cell.output.items_count !== 1 ? 's' : ''}
            </span>
            {cell.output.operator_stats && (
              <span className="text-[11px] text-gray-400">
                · {cell.output.operator_stats.wall_clock_secs?.toFixed(1)}s
                · ${cell.output.operator_stats.total_cost_usd?.toFixed(4) || '0.00'}
              </span>
            )}
          </div>

          {/* Answer text for reasoning cells */}
          {cell.output.answer && (
            <div className="mb-2 text-sm text-gray-700 bg-white p-2 rounded border border-gray-200 whitespace-pre-wrap">
              {cell.output.answer}
            </div>
          )}

          {/* Table preview */}
          {cell.output.preview && cell.output.preview.length > 0 && cell.output.schema && (
            <div className="overflow-x-auto rounded border border-gray-200">
              <table className="text-[11px] w-full">
                <thead>
                  <tr className="bg-gray-100">
                    {cell.output.schema.map((col) => (
                      <th key={col} className="px-2 py-1 text-left font-semibold text-gray-600 border-b border-gray-200">
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {cell.output.preview.map((row, rowIdx) => (
                    <tr key={rowIdx} className={rowIdx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                      {cell.output.schema.map((col) => (
                        <td key={col} className="px-2 py-1 text-gray-700 border-b border-gray-100 max-w-[200px] truncate">
                          {typeof row[col] === 'object' ? JSON.stringify(row[col]) : String(row[col] ?? '')}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
              {cell.output.items_count > 10 && (
                <div className="text-center py-1 text-[10px] text-gray-400 bg-gray-50 border-t border-gray-200">
                  … and {cell.output.items_count - 10} more rows
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Error output */}
      {cell.status === 'error' && cell.error && (
        <div className="border-t border-gray-200 p-3 bg-red-50">
          <div className="flex items-center gap-2 text-red-700 text-sm">
            <AlertCircle className="w-4 h-4 flex-shrink-0" />
            <span>{cell.error}</span>
          </div>
        </div>
      )}
    </div>
  )
}

export default NotebookCell
