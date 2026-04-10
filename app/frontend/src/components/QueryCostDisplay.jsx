import { DollarSign } from 'lucide-react'

/**
 * QueryCostDisplay
 *
 * A compact, read-only pill that shows the running cost of the current
 * query.  Appears next to the CostBudgetPicker so the user can compare
 * spend-so-far against their budget at a glance.
 *
 * The pill turns:
 *   - green  when cost < 50 % of budget
 *   - amber  when 50 % ≤ cost < 90 % of budget
 *   - red    when cost ≥ 90 % of budget
 *
 * Props:
 *   cost   – current cumulative cost in USD (number | null)
 *   budget – the user's cost budget in USD (number)
 */
export default function QueryCostDisplay({ cost, budget }) {
  if (cost == null) return null

  const ratio = budget > 0 ? cost / budget : 0
  let colorClasses
  if (ratio >= 0.9) {
    colorClasses = 'bg-red-50 border-red-300 text-red-600'
  } else if (ratio >= 0.5) {
    colorClasses = 'bg-amber-50 border-amber-300 text-amber-600'
  } else {
    colorClasses = 'bg-green-50 border-green-300 text-green-600'
  }

  const display = cost < 0.001 ? '<0.001' : cost < 10 ? cost.toFixed(3) : cost.toFixed(2)

  return (
    <div
      title={`Query cost so far: $${cost.toFixed(4)} of $${budget} budget`}
      className={`
        flex items-center gap-1 px-3 py-2 rounded-xl border text-sm font-medium
        select-none transition-colors ${colorClasses}
      `}
    >
      <DollarSign className="w-3.5 h-3.5" />
      <span>{display}</span>
    </div>
  )
}
