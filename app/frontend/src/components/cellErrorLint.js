/**
 * CodeMirror 6 lint source that surfaces cell execution errors as
 * inline diagnostics.
 *
 * The backend returns a single error string on cell failure.  We
 * display this as a "hint" diagnostic anchored to the first line of
 * the editor, using CodeMirror's lint gutter.
 *
 * Representation invariant:
 *   - ``makeCellDiagnostics`` returns an array of zero or one
 *     ``Diagnostic`` objects.
 *
 * Abstraction function:
 *   Represents a bridge between backend execution errors and the
 *   CodeMirror diagnostic/lint UI.  When an error string is non-empty,
 *   it appears as a red marker in the gutter; when null/empty, the
 *   gutter is clean.
 */
import { linter, lintGutter } from '@codemirror/lint'
import { StateEffect, StateField } from '@codemirror/state'

/* ------------------------------------------------------------------ */
/*  State field: the current error string (or null)                    */
/* ------------------------------------------------------------------ */

/** Effect to push a new error (or clear it with null). */
export const setCellError = StateEffect.define()

/** State field that holds the latest error string for this editor. */
const cellErrorField = StateField.define({
  create: () => null,
  update(value, tr) {
    for (const e of tr.effects) {
      if (e.is(setCellError)) return e.value
    }
    return value
  },
})

/* ------------------------------------------------------------------ */
/*  Linter that reads from the state field                             */
/* ------------------------------------------------------------------ */

const cellErrorLinter = linter((view) => {
  const error = view.state.field(cellErrorField)
  if (!error) return []

  // Anchor the diagnostic to the full first line.
  const firstLine = view.state.doc.line(1)
  return [{
    from: firstLine.from,
    to: firstLine.to,
    severity: 'error',
    message: error,
  }]
})

/* ------------------------------------------------------------------ */
/*  Public API                                                         */
/* ------------------------------------------------------------------ */

/**
 * Returns a CodeMirror extension array that enables the error gutter
 * and a linter driven by ``setCellError`` effects.
 *
 * Usage from React:
 *   1. Include ``cellErrorExtension()`` in the editor's extensions.
 *   2. When a cell error arrives, dispatch:
 *      ``view.dispatch({ effects: setCellError.of(errorString) })``
 *   3. When the cell re-runs or succeeds, dispatch:
 *      ``view.dispatch({ effects: setCellError.of(null) })``
 *
 * Requires:
 *   None.
 *
 * Returns:
 *   An array of CodeMirror extensions: ``[cellErrorField, lintGutter, cellErrorLinter]``.
 *
 * Raises:
 *   None.
 */
export function cellErrorExtension() {
  return [cellErrorField, lintGutter(), cellErrorLinter]
}
