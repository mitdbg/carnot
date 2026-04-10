import { X, MessageSquare, FileCode } from 'lucide-react'

/**
 * A simple tab bar for the workspace. Each tab has a label, an icon,
 * and an optional close button (the chat tab is not closeable).
 *
 * Props:
 *   tabs       – [{ id, label, type: 'chat' | 'notebook' }]
 *   activeId   – the currently active tab id
 *   onSelect   – (tabId) => void
 *   onClose    – (tabId) => void
 */
function TabBar({ tabs, activeId, onSelect, onClose }) {
  return (
    <div className="flex items-center border-b border-gray-200 bg-gray-50 px-2 overflow-x-auto flex-shrink-0">
      {tabs.map((tab) => {
        const isActive = tab.id === activeId
        const Icon = tab.type === 'notebook' ? FileCode : MessageSquare
        return (
          <button
            key={tab.id}
            onClick={() => onSelect(tab.id)}
            className={`
              group flex items-center gap-1.5 px-3 py-2 text-sm font-medium
              border-b-2 transition-colors whitespace-nowrap
              ${isActive
                ? 'border-primary-500 text-primary-600 bg-white'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:bg-gray-100'
              }
            `}
          >
            <Icon className="w-3.5 h-3.5" />
            <span className="max-w-[140px] truncate">{tab.label}</span>
            {/* Close button — only for notebook tabs */}
            {tab.type === 'notebook' && (
              <span
                onClick={(e) => { e.stopPropagation(); onClose(tab.id); }}
                className="ml-1 p-0.5 rounded hover:bg-gray-200 opacity-0 group-hover:opacity-100 transition-opacity"
              >
                <X className="w-3 h-3" />
              </span>
            )}
          </button>
        )
      })}
    </div>
  )
}

export default TabBar
