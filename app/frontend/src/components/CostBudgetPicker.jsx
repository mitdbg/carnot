import { useState, useRef, useEffect } from 'react'
import { DollarSign, ChevronDown } from 'lucide-react'

const PRESET_OPTIONS = [1, 2, 5, 10, 20]

/**
 * CostBudgetPicker
 *
 * Displays a compact pill showing the current budget (e.g. "$5").
 * When clicked, a horizontal popover opens above it with preset dollar
 * options ($1 $2 $5 $10 $20) plus a "Custom" option.  Selecting "Custom"
 * replaces the popover with a small inline input that accepts a float with
 * up to two decimal places.
 *
 * Props:
 *   value    – current budget (number)
 *   onChange – callback(newValue: number)
 *   disabled – greys out the control when the query is loading
 */
export default function CostBudgetPicker({ value, onChange, disabled }) {
  const [open, setOpen] = useState(false)
  const [customMode, setCustomMode] = useState(false)
  const [customInput, setCustomInput] = useState('')
  const containerRef = useRef(null)
  const customInputRef = useRef(null)

  // Close the popover when the user clicks outside
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (containerRef.current && !containerRef.current.contains(e.target)) {
        closeAll()
      }
    }
    if (open) document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [open])

  // Focus the custom input as soon as it appears
  useEffect(() => {
    if (customMode && customInputRef.current) {
      customInputRef.current.focus()
    }
  }, [customMode])

  const closeAll = () => {
    setOpen(false)
    setCustomMode(false)
    setCustomInput('')
  }

  const handlePresetClick = (amount) => {
    onChange(amount)
    closeAll()
  }

  const handleCustomConfirm = () => {
    const parsed = parseFloat(parseFloat(customInput).toFixed(2))
    if (!isNaN(parsed) && parsed > 0) {
      onChange(parsed)
    }
    closeAll()
  }

  const handleCustomKeyDown = (e) => {
    if (e.key === 'Enter') handleCustomConfirm()
    if (e.key === 'Escape') closeAll()
  }

  // Format the display value: show as integer if whole number, else 2 dp
  const displayValue = Number.isInteger(value) ? `$${value}` : `$${value.toFixed(2)}`

  return (
    <div ref={containerRef} className="relative flex-shrink-0">
      {/* ── Pill button ── */}
      <button
        type="button"
        onClick={() => { if (!disabled) setOpen((o) => !o) }}
        disabled={disabled}
        title="Set maximum cost budget for this query"
        className={`
          flex items-center gap-1 px-3 py-2 rounded-xl border text-sm font-medium
          transition-colors select-none
          ${disabled
            ? 'bg-gray-100 border-gray-200 text-gray-400 cursor-not-allowed'
            : 'bg-white border-gray-300 text-gray-700 hover:border-primary-400 hover:text-primary-600 cursor-pointer'}
          ${open ? 'border-primary-400 text-primary-600 ring-2 ring-primary-200' : ''}
        `}
      >
        <DollarSign className="w-3.5 h-3.5" />
        <span>{displayValue.slice(1)}</span>
        <ChevronDown className={`w-3 h-3 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>

      {/* ── Popover ── */}
      {open && (
        <div
          className="
            absolute bottom-full mb-2 right-0
            bg-white border border-gray-200 rounded-xl shadow-lg
            p-2 z-50
            min-w-max
          "
        >
          {!customMode ? (
            /* ── Preset row ── */
            <div className="flex items-center gap-1.5">
              {PRESET_OPTIONS.map((amount) => (
                <button
                  key={amount}
                  type="button"
                  onClick={() => handlePresetClick(amount)}
                  className={`
                    px-3 py-1.5 rounded-lg text-sm font-medium transition-colors
                    ${value === amount
                      ? 'bg-primary-500 text-white'
                      : 'bg-gray-100 text-gray-700 hover:bg-primary-50 hover:text-primary-700'}
                  `}
                >
                  ${amount}
                </button>
              ))}

              {/* Divider */}
              <div className="w-px h-5 bg-gray-200 mx-0.5" />

              {/* Custom option */}
              <button
                type="button"
                onClick={() => {
                  setCustomMode(true)
                  setCustomInput('')
                }}
                className="
                  px-3 py-1.5 rounded-lg text-sm font-medium transition-colors
                  bg-gray-100 text-gray-700 hover:bg-primary-50 hover:text-primary-700
                "
              >
                Custom
              </button>
            </div>
          ) : (
            /* ── Custom input ── */
            <div className="flex items-center gap-2 px-1">
              <span className="text-xs text-gray-500 font-medium whitespace-nowrap">Custom budget:</span>
              <div className="flex items-center border border-gray-300 rounded-lg overflow-hidden focus-within:ring-2 focus-within:ring-primary-400 focus-within:border-primary-400">
                <span className="pl-2 text-gray-400 text-sm select-none">$</span>
                <input
                  ref={customInputRef}
                  type="number"
                  min="0.01"
                  step="0.01"
                  placeholder="0.00"
                  value={customInput}
                  onChange={(e) => setCustomInput(e.target.value)}
                  onKeyDown={handleCustomKeyDown}
                  className="w-24 py-1.5 px-1.5 text-sm text-gray-800 border-none focus:ring-0 bg-transparent"
                />
              </div>
              <button
                type="button"
                onClick={handleCustomConfirm}
                disabled={!customInput || isNaN(parseFloat(customInput)) || parseFloat(customInput) <= 0}
                className="px-3 py-1.5 rounded-lg text-sm font-medium bg-primary-500 text-white hover:bg-primary-600 disabled:bg-gray-200 disabled:text-gray-400 transition-colors"
              >
                Set
              </button>
              <button
                type="button"
                onClick={closeAll}
                className="px-2 py-1.5 rounded-lg text-sm text-gray-500 hover:bg-gray-100 transition-colors"
              >
                Cancel
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
