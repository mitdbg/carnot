/**
 * CodeMirror 6 autocomplete source for Carnot notebook cells.
 *
 * Provides context-aware completions for:
 * - Carnot API functions (sem_filter, sem_map, sem_join, etc.)
 * - Common variables (datasets, carnot)
 * - Python builtins frequently used in cells (print, len, range, etc.)
 *
 * Representation invariant:
 *   - Every entry in COMPLETIONS has at least `label` and `type`.
 *   - `boost` values are higher for Carnot-specific completions so
 *     they sort above generic Python builtins.
 *
 * Abstraction function:
 *   Represents a static dictionary of completions that the CodeMirror
 *   autocompletion extension queries on each keystroke.
 */
import { autocompletion } from '@codemirror/autocomplete'

/* ------------------------------------------------------------------ */
/*  Completion entries                                                 */
/* ------------------------------------------------------------------ */

const CARNOT_FUNCTIONS = [
  {
    label: 'sem_filter',
    type: 'function',
    detail: '(dataset, condition)',
    info: 'Semantic filter — keeps rows matching a natural-language condition.',
    boost: 10,
  },
  {
    label: 'sem_map',
    type: 'function',
    detail: '(dataset, field, type, description)',
    info: 'Semantic map — adds a computed field to each row.',
    boost: 10,
  },
  {
    label: 'sem_flat_map',
    type: 'function',
    detail: '(dataset, field, type)',
    info: 'Semantic flat map — one-to-many mapping, expanding rows.',
    boost: 10,
  },
  {
    label: 'sem_join',
    type: 'function',
    detail: '(left, right, condition)',
    info: 'Semantic join — joins two datasets on a natural-language condition.',
    boost: 10,
  },
  {
    label: 'sem_topk',
    type: 'function',
    detail: '(dataset, search, k)',
    info: 'Semantic top-K — retrieves the k most relevant rows.',
    boost: 10,
  },
  {
    label: 'sem_group_by',
    type: 'function',
    detail: '(dataset, group_by, aggregations)',
    info: 'Semantic group by — groups rows and applies aggregations.',
    boost: 10,
  },
  {
    label: 'sem_agg',
    type: 'function',
    detail: '(dataset, agg_fields)',
    info: 'Semantic aggregation — computes aggregate values over a dataset.',
    boost: 10,
  },
  {
    label: 'code_operator',
    type: 'function',
    detail: '(task, datasets)',
    info: 'Code operator — executes arbitrary Python on all datasets.',
    boost: 8,
  },
  {
    label: 'reasoning_operator',
    type: 'function',
    detail: '(query, datasets)',
    info: 'Reasoning operator — synthesises a final answer from datasets.',
    boost: 8,
  },
  {
    label: 'limit',
    type: 'function',
    detail: '(dataset, n)',
    info: 'Limit — returns the first n rows of a dataset.',
    boost: 8,
  },
  {
    label: 'carnot.load_dataset',
    type: 'function',
    detail: '(name)',
    info: 'Load a named dataset into the notebook.',
    boost: 9,
  },
]

const CARNOT_VARIABLES = [
  {
    label: 'datasets',
    type: 'variable',
    detail: 'dict[str, Dataset]',
    info: 'The shared datasets dict — keys are dataset IDs, values are Dataset objects.',
    boost: 7,
  },
  {
    label: 'carnot',
    type: 'namespace',
    detail: 'module',
    info: 'The carnot Python package.',
    boost: 6,
  },
]

const PYTHON_BUILTINS = [
  { label: 'print', type: 'function', detail: '(*args)', boost: 1 },
  { label: 'len', type: 'function', detail: '(obj)', boost: 1 },
  { label: 'range', type: 'function', detail: '(stop)', boost: 1 },
  { label: 'list', type: 'function', detail: '(iterable)', boost: 1 },
  { label: 'dict', type: 'function', detail: '()', boost: 1 },
  { label: 'str', type: 'function', detail: '(obj)', boost: 1 },
  { label: 'int', type: 'function', detail: '(x)', boost: 1 },
  { label: 'float', type: 'function', detail: '(x)', boost: 1 },
  { label: 'sorted', type: 'function', detail: '(iterable)', boost: 1 },
  { label: 'enumerate', type: 'function', detail: '(iterable)', boost: 1 },
  { label: 'zip', type: 'function', detail: '(*iterables)', boost: 1 },
  { label: 'isinstance', type: 'function', detail: '(obj, classinfo)', boost: 1 },
  { label: 'True', type: 'keyword', boost: 0 },
  { label: 'False', type: 'keyword', boost: 0 },
  { label: 'None', type: 'keyword', boost: 0 },
  { label: 'import', type: 'keyword', boost: 0 },
  { label: 'from', type: 'keyword', boost: 0 },
  { label: 'return', type: 'keyword', boost: 0 },
  { label: 'for', type: 'keyword', boost: 0 },
  { label: 'if', type: 'keyword', boost: 0 },
  { label: 'else', type: 'keyword', boost: 0 },
  { label: 'elif', type: 'keyword', boost: 0 },
  { label: 'while', type: 'keyword', boost: 0 },
  { label: 'def', type: 'keyword', boost: 0 },
  { label: 'class', type: 'keyword', boost: 0 },
]

const ALL_COMPLETIONS = [
  ...CARNOT_FUNCTIONS,
  ...CARNOT_VARIABLES,
  ...PYTHON_BUILTINS,
]

/* ------------------------------------------------------------------ */
/*  Completion source                                                  */
/* ------------------------------------------------------------------ */

/**
 * CodeMirror completion source that matches against the static
 * Carnot + Python vocabulary.
 *
 * Requires:
 *   Called by the CodeMirror autocompletion extension; receives a
 *   ``CompletionContext`` with the current editor state.
 *
 * Returns:
 *   ``{ from, options }`` when the cursor follows a word-like prefix
 *   of ≥ 2 characters, or ``null`` otherwise.
 *
 * Raises:
 *   None.
 */
function carnotCompletionSource(context) {
  // Match word chars plus dots (for `carnot.load_dataset`)
  const word = context.matchBefore(/[\w.]+/)
  if (!word || word.from === word.to) return null
  // Only trigger after typing at least 2 characters
  if (word.to - word.from < 2 && !context.explicit) return null

  return {
    from: word.from,
    options: ALL_COMPLETIONS,
    validFor: /^[\w.]*$/,
  }
}

/* ------------------------------------------------------------------ */
/*  Public extension                                                   */
/* ------------------------------------------------------------------ */

/**
 * A CodeMirror extension that enables Carnot-aware autocompletion.
 *
 * Requires:
 *   None.
 *
 * Returns:
 *   A CodeMirror ``Extension`` to include in the editor's extension
 *   array.
 *
 * Raises:
 *   None.
 */
export function carnotAutocomplete() {
  return autocompletion({
    override: [carnotCompletionSource],
    activateOnTyping: true,
    maxRenderedOptions: 15,
  })
}
