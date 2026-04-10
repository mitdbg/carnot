import { useState, useRef, useCallback, useEffect } from 'react'
import { Play, Square, Loader2, Plus } from 'lucide-react'
import { useApiToken } from '../hooks/useApiToken'
import { notebookApi } from '../services/api'
import NotebookCell from './NotebookCell'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api"

const OPERATOR_TYPES = [
  { label: 'Semantic Filter', value: 'SemanticFilter' },
  { label: 'Semantic Map', value: 'SemanticMap' },
  { label: 'Semantic Join', value: 'SemanticJoin' },
  { label: 'Semantic Top-K', value: 'SemanticTopK' },
  { label: 'Semantic GroupBy', value: 'SemanticGroupBy' },
  { label: 'Semantic Agg', value: 'SemanticAgg' },
  { label: 'Code', value: 'Code' },
  { label: 'Limit', value: 'Limit' },
  { label: 'Blank cell', value: null },
]

/**
 * Renders an interactive notebook for a single query execution.
 * Each cell corresponds to one physical operator in the plan.
 *
 * Props:
 *   notebook     – { notebookId, query, cells }
 *   onUpdate     – (notebookId, updatedNotebook) => void  — called when cell state changes
 *
 * Representation invariant:
 *   - `cellsRef.current` always mirrors `notebook.cells` (kept in sync
 *     via a `useEffect`).
 *   - All cell-mutation helpers (`updateCells`, `replaceCells`) read from
 *     `cellsRef` so async code (SSE handlers, Run All loop) never sees
 *     stale closures.
 *
 * Abstraction function:
 *   Represents the running state of a notebook: which cell is executing,
 *   whether a Run All sweep is in progress, and the latest cells array.
 */
function NotebookView({ notebook, onUpdate }) {
  const getValidToken = useApiToken()
  const [runningCellId, setRunningCellId] = useState(null)
  const [isRunningAll, setIsRunningAll] = useState(false)
  const stopRef = useRef(false)
  const [addMenuCellId, setAddMenuCellId] = useState(null) // which "+" button is open

  /* ---------- stale-closure guard ----------
   * `notebook.cells` comes from props and is frozen at render time.
   * During Run All the async loop holds a single closure for many
   * sequential awaits, so by the time cell N+1 starts, `notebook.cells`
   * still reflects the state *before* cell N completed — causing each
   * updateCell call to overwrite previous outputs.
   *
   * Fix: keep a ref that is always up-to-date and have every mutation
   * read from the ref instead of the prop.
   */
  const cellsRef = useRef(notebook.cells)
  useEffect(() => { cellsRef.current = notebook.cells }, [notebook.cells])

  const emitUpdate = useCallback((cells) => {
    cellsRef.current = cells
    onUpdate(notebook.notebookId, { notebookId: notebook.notebookId, query: notebook.query, cells })
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [notebook.notebookId, notebook.query, onUpdate])

  const updateCell = useCallback((cellId, updates) => {
    const updatedCells = cellsRef.current.map((c) =>
      c.cell_id === cellId ? { ...c, ...updates } : c
    )
    cellsRef.current = updatedCells
    onUpdate(notebook.notebookId, { notebookId: notebook.notebookId, query: notebook.query, cells: updatedCells })
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [notebook.notebookId, notebook.query, onUpdate])

  const replaceCells = useCallback((cells) => {
    cellsRef.current = cells
    onUpdate(notebook.notebookId, { notebookId: notebook.notebookId, query: notebook.query, cells })
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [notebook.notebookId, notebook.query, onUpdate])

  const runCell = useCallback(async (cellId, code = null) => {
    const token = await getValidToken()
    if (!token) return

    setRunningCellId(cellId)
    updateCell(cellId, { status: 'running' })

    try {
      const body = {
        notebook_id: notebook.notebookId,
        cell_id: cellId,
      }
      if (code !== null) {
        body.code = code
      }

      const response = await fetch(`${API_BASE_URL}/query/execute-cell`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify(body),
      })

      if (!response.ok) {
        const err = await response.json().catch(() => ({}))
        throw new Error(err.detail || `Server responded with ${response.status}`)
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

              if (data.type === 'cell_complete') {
                // Handle downstream invalidation together with completion
                // so we do a single atomic update from the latest cells.
                if (data.invalidated_cells && data.invalidated_cells.length > 0) {
                  const merged = cellsRef.current.map((c) => {
                    if (c.cell_id === cellId) {
                      return { ...c, status: 'success', output: data.output }
                    }
                    if (data.invalidated_cells.includes(c.cell_id)) {
                      return { ...c, status: 'pending', output: undefined }
                    }
                    return c
                  })
                  cellsRef.current = merged
                  onUpdate(notebook.notebookId, { notebookId: notebook.notebookId, query: notebook.query, cells: merged, active: true })
                } else {
                  const updatedCells = cellsRef.current.map((c) =>
                    c.cell_id === cellId ? { ...c, status: 'success', output: data.output } : c
                  )
                  cellsRef.current = updatedCells
                  onUpdate(notebook.notebookId, { notebookId: notebook.notebookId, query: notebook.query, cells: updatedCells, active: true })
                }
              } else if (data.type === 'cell_error') {
                updateCell(cellId, {
                  status: 'error',
                  error: data.error,
                })
              }
            } catch (e) {
              console.error('Error parsing SSE:', e)
            }
          }
        }
      }
    } catch (error) {
      console.error('Cell execution failed:', error)
      updateCell(cellId, {
        status: 'error',
        error: error.message || 'Cell execution failed',
      })
    } finally {
      setRunningCellId(null)
    }
  }, [getValidToken, notebook.notebookId, notebook.query, onUpdate, updateCell])

  const runAllCells = async () => {
    setIsRunningAll(true)
    stopRef.current = false

    // Read cell IDs from the ref so we always see the latest list
    const cellIds = cellsRef.current.map((c) => ({ id: c.cell_id, status: c.status }))
    for (const { id, status } of cellIds) {
      if (stopRef.current) break
      // Skip already-succeeded cells on re-run-all — but only when the
      // kernel is alive.  After a restart (active === false) the in-memory
      // datasets_store is gone, so every cell must be re-executed even if
      // its persisted status is "success".
      if (notebook.active && status === 'success') continue
      await runCell(id)
    }

    setIsRunningAll(false)
  }

  const stopRunAll = () => {
    stopRef.current = true
  }

  const handleDeleteCell = async (cellId) => {
    try {
      const token = await getValidToken()
      if (!token) return
      const response = await notebookApi.deleteCell(notebook.notebookId, cellId, token)
      replaceCells(response.data.updated_cells)
    } catch (error) {
      console.error('Error deleting cell:', error)
    }
  }

  const handleAddCell = async (afterCellId, operatorType) => {
    setAddMenuCellId(null)
    try {
      const token = await getValidToken()
      if (!token) return
      const response = await notebookApi.addCell(notebook.notebookId, {
        after_cell_id: afterCellId,
        cell_type: 'operator',
        operator_type: operatorType,
      }, token)
      replaceCells(response.data.updated_cells)
    } catch (error) {
      console.error('Error adding cell:', error)
    }
  }

  const handleMoveCell = async (cellId, direction) => {
    try {
      const token = await getValidToken()
      if (!token) return
      const response = await notebookApi.moveCell(notebook.notebookId, {
        cell_id: cellId,
        direction,
      }, token)
      replaceCells(response.data.updated_cells)
    } catch (error) {
      console.error('Error moving cell:', error)
    }
  }

  const completedCount = notebook.cells.filter((c) => c.status === 'success').length
  const totalCount = notebook.cells.length
  const isExpired = notebook.active === false

  return (
    <div className="flex-1 flex flex-col min-h-0">
      {/* Session expired banner */}
      {isExpired && (
        <div className="px-6 py-2 bg-amber-50 border-b border-amber-200 text-amber-800 text-sm flex items-center gap-2">
          <span className="font-medium">⚠ Notebook session expired</span>
          <span className="text-amber-600">— kernel state has been evicted. Re-run cells from the top to execute new operations.</span>
        </div>
      )}

      {/* Toolbar */}
      <div className="flex items-center justify-between px-6 py-3 border-b border-gray-200 bg-white flex-shrink-0">
        <div className="flex items-center gap-3">
          <h3 className="text-sm font-semibold text-gray-700 truncate max-w-md">
            {notebook.query}
          </h3>
          <span className="text-[10px] text-gray-400 font-mono">
            {completedCount}/{totalCount} cells
          </span>
        </div>
        <div className="flex items-center gap-2">
          {isRunningAll ? (
            <button
              onClick={stopRunAll}
              className="flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium rounded-lg
                bg-red-600 hover:bg-red-700 text-white transition-colors"
            >
              <Square className="w-3.5 h-3.5" />
              Stop
            </button>
          ) : (
            <button
              onClick={runAllCells}
              disabled={runningCellId !== null}
              className="flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium rounded-lg
                bg-green-600 hover:bg-green-700 text-white
                disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
            >
              {runningCellId !== null
                ? <Loader2 className="w-3.5 h-3.5 animate-spin" />
                : <Play className="w-3.5 h-3.5" />
              }
              Run All
            </button>
          )}
        </div>
      </div>

      {/* Cell List */}
      <div className="flex-1 overflow-y-auto px-6 py-4">
        <div className="max-w-4xl mx-auto space-y-1">
          {notebook.cells.map((cell, idx) => (
            <div key={cell.cell_id}>
              <NotebookCell
                cell={cell}
                index={idx}
                onRun={runCell}
                onDelete={handleDeleteCell}
                onMove={handleMoveCell}
                isRunning={runningCellId === cell.cell_id}
                disabled={runningCellId !== null && runningCellId !== cell.cell_id}
                isFirst={idx === 0}
                isLast={idx === notebook.cells.length - 1}
                canDelete={cell.cell_type !== 'dataset' && cell.cell_type !== 'reasoning'}
              />
              {/* Add Cell Button between cells */}
              {idx < notebook.cells.length - 1 && (
                <div className="relative flex justify-center py-1">
                  <button
                    onClick={() => setAddMenuCellId(addMenuCellId === cell.cell_id ? null : cell.cell_id)}
                    className="group flex items-center gap-1 px-2 py-0.5 text-[10px] text-gray-400
                      hover:text-primary-600 hover:bg-primary-50 rounded transition-colors"
                  >
                    <Plus className="w-3 h-3" />
                    <span className="opacity-0 group-hover:opacity-100 transition-opacity">Add cell</span>
                  </button>
                  {/* Dropdown menu */}
                  {addMenuCellId === cell.cell_id && (
                    <div className="absolute top-full z-20 bg-white border border-gray-200 rounded-lg shadow-lg py-1 min-w-[180px]">
                      {OPERATOR_TYPES.map(({ label, value }) => (
                        <button
                          key={label}
                          onClick={() => handleAddCell(cell.cell_id, value)}
                          className="w-full text-left px-3 py-1.5 text-sm text-gray-700 hover:bg-primary-50
                            hover:text-primary-700 transition-colors"
                        >
                          + {label}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export default NotebookView
