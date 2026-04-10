import { useMemo, useRef, useEffect } from 'react'
import CodeMirror from '@uiw/react-codemirror'
import { python } from '@codemirror/lang-python'
import { vscodeDark } from '@uiw/codemirror-theme-vscode'
import { keymap } from '@codemirror/view'
import { indentWithTab } from '@codemirror/commands'
import { EditorView } from '@codemirror/view'
import { foldGutter } from '@codemirror/language'
import { carnotAutocomplete } from './carnotCompletions'
import { cellErrorExtension, setCellError } from './cellErrorLint'

/**
 * Editable code block with Python syntax highlighting, powered by
 * CodeMirror 6.
 *
 * Features:
 *   - Python syntax highlighting (VS Code Dark theme)
 *   - Carnot-aware autocomplete (sem_filter, sem_map, etc.)
 *   - Inline error diagnostics from cell execution failures
 *   - Code folding for large cells
 *   - Shift+Enter to run the cell
 *   - Tab-indent, bracket matching, close-brackets, undo/redo
 *
 * Props:
 *   code      – the current code string
 *   onChange  – (newCode: string) => void
 *   readOnly  – if true, disable editing
 *   onRun     – optional () => void; invoked on Shift+Enter
 *   error     – optional string; when non-null, shown as a lint diagnostic
 *
 * Representation invariant:
 *   - `extensions` is recomputed when `onRun` changes.
 *   - The component is fully controlled: `code` is the source of truth
 *     and `onChange` is called on every keystroke.
 *   - When `error` transitions from null → string (or vice versa), a
 *     `setCellError` effect is dispatched into the editor.
 *
 * Abstraction function:
 *   Represents a single editable (or read-only) Python code cell whose
 *   visual appearance matches VS Code Dark and whose content is always
 *   synchronised with the parent's state via `code` / `onChange`.
 */

/* Let the editor grow to fit content instead of using a fixed height.
   Also sets our desired font metrics to match the rest of the notebook. */
const baseTheme = EditorView.theme({
  '&': {
    fontSize: '0.8125rem',
    backgroundColor: '#1e1e1e',
  },
  '.cm-content': {
    fontFamily: 'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace',
    lineHeight: '1.6',
    padding: '1rem 0',
  },
  '.cm-gutters': {
    backgroundColor: '#1e1e1e',
    border: 'none',
  },
  '.cm-scroller': {
    overflow: 'auto',
  },
  /* Dim the editor when read-only so the user sees it's not editable */
  '&.cm-editor.cm-readonly': {
    opacity: '0.65',
  },
  /* Hide the cursor in read-only mode */
  '&.cm-editor.cm-readonly .cm-cursor': {
    display: 'none',
  },
  /* Soften the active-line highlight */
  '.cm-activeLine': {
    backgroundColor: 'rgba(255,255,255,0.04)',
  },
  '.cm-selectionBackground': {
    backgroundColor: 'rgba(59,130,246,0.35) !important',
  },
  /* Fold gutter styling */
  '.cm-foldGutter span': {
    color: '#6b7280',
    fontSize: '0.75rem',
  },
  /* Lint gutter: make error dots visible against dark background */
  '.cm-lint-marker-error': {
    content: 'none',
  },
  '.cm-diagnostic-error': {
    borderLeft: '3px solid #ef4444',
    backgroundColor: 'rgba(239,68,68,0.08)',
    padding: '4px 8px',
    marginTop: '2px',
    fontSize: '0.75rem',
  },
  /* Autocomplete dropdown styling */
  '.cm-tooltip-autocomplete': {
    backgroundColor: '#252526',
    border: '1px solid #3c3c3c',
  },
  '.cm-tooltip-autocomplete ul li': {
    color: '#cccccc',
  },
  '.cm-tooltip-autocomplete ul li[aria-selected]': {
    backgroundColor: '#094771',
    color: '#ffffff',
  },
  '.cm-completionLabel': {
    fontSize: '0.8125rem',
  },
  '.cm-completionDetail': {
    color: '#888888',
    fontSize: '0.75rem',
    fontStyle: 'italic',
  },
})

function CodeEditor({ code, onChange, readOnly = false, onRun, error = null }) {
  const editorViewRef = useRef(null)

  /* Dispatch error changes into the CM state so the linter picks them up. */
  useEffect(() => {
    const view = editorViewRef.current
    if (view) {
      view.dispatch({ effects: setCellError.of(error || null) })
    }
  }, [error])

  const extensions = useMemo(
    () => {
      const exts = [
        python(),
        keymap.of([indentWithTab]),
        baseTheme,
        EditorView.lineWrapping,
        foldGutter(),
        carnotAutocomplete(),
        cellErrorExtension(),
      ]

      // Shift+Enter → run the cell
      if (onRun) {
        exts.push(
          keymap.of([{
            key: 'Shift-Enter',
            run: () => { onRun(); return true },
          }])
        )
      }

      return exts
    },
    [onRun]
  )

  return (
    <CodeMirror
      value={code}
      onChange={onChange}
      readOnly={readOnly}
      theme={vscodeDark}
      extensions={extensions}
      basicSetup={{
        lineNumbers: false,
        foldGutter: false,        // we add our own above
        highlightActiveLine: !readOnly,
        bracketMatching: true,
        closeBrackets: true,
        autocompletion: false,    // we add our own above
      }}
      onCreateEditor={(view) => { editorViewRef.current = view }}
    />
  )
}

export default CodeEditor
