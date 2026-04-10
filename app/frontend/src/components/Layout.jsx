import { Link, useLocation } from 'react-router-dom'
import { Database, MessageSquare, LogOut, Settings } from 'lucide-react'
import { useAuth0 } from '@auth0/auth0-react'

function Layout({ children }) {
  const location = useLocation()
  const { logout } = useAuth0()

  const tabs = [
    { name: 'Data Management', path: '/', icon: Database },
    { name: 'Workspace', path: '/chat', icon: MessageSquare },
    { name: 'Settings', path: '/settings', icon: Settings },
  ]

  return (
    /* h-screen + overflow-hidden prevents the whole page from scrolling */
    <div className="h-screen w-full flex flex-col bg-white overflow-hidden">
      
      {/* Header: Removed max-w-7xl to make it flush */}
      <header className="flex-shrink-0 bg-white border-b border-gray-200">
        <div className="px-6 flex items-center justify-between h-14">
          <div className="flex items-center">
            <h1 className="text-xl font-bold text-primary-600">Carnot</h1>
            <span className="ml-2 text-xs text-gray-400 font-medium uppercase tracking-wider">
              Deep Research Engine
            </span>
          </div>
        </div>
      </header>

      {/* Tabs + Logout: Removed max-w-7xl and side padding */}
      <div className="flex-shrink-0 bg-white border-b border-gray-200 px-6">
        <nav className="flex items-center justify-between" aria-label="Tabs">
          <div className="flex space-x-6">
            {tabs.map((tab) => {
              const isActive = location.pathname === tab.path
              const Icon = tab.icon
              return (
                <Link
                  key={tab.name}
                  to={tab.path}
                  className={`
                    flex items-center gap-2 py-3 px-1 border-b-2 font-medium text-sm
                    transition-colors duration-200
                    ${isActive
                        ? 'border-primary-500 text-primary-600'
                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                    }
                  `}
                >
                  <Icon className="w-4 h-4" />
                  {tab.name}
                </Link>
              )
            })}
          </div>

          <button
            onClick={() => logout({ logoutParams: { returnTo: window.location.origin } })}
            className="flex items-center gap-2 py-2 text-sm font-medium text-gray-500 hover:text-red-600 transition-colors"
          >
            <LogOut className="w-4 h-4" />
            Logout
          </button>
        </nav>
      </div>

      {/* Main Content: flex-1 takes up all remaining vertical space.
          Removed py-8 and max-w-7xl so children can be flush. */}
      <main className="flex-1 min-h-0 relative overflow-y-auto">
        {children}
      </main>
    </div>
  )
}

export default Layout

// import { Link, useLocation } from 'react-router-dom'
// import { Database, MessageSquare, LogOut, Settings } from 'lucide-react'
// import { useAuth0 } from '@auth0/auth0-react'

// function Layout({ children }) {
//   const location = useLocation()
//   const { logout } = useAuth0()

//   const tabs = [
//     { name: 'Data Management', path: '/', icon: Database },
//     { name: 'User Chat', path: '/chat', icon: MessageSquare },
//     { name: 'Settings', path: '/settings', icon: Settings },
//   ]

//   return (
//     <div className="min-h-screen bg-gray-50">
//       {/* Header */}
//       <header className="bg-white border-b border-gray-200 shadow-sm">
//         <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
//           <div className="flex items-center justify-between h-16">
//             <div className="flex items-center">
//               <h1 className="text-2xl font-bold text-primary-600">Carnot</h1>
//               <span className="ml-2 text-sm text-gray-500">Deep Research Engine</span>
//             </div>
//           </div>
//         </div>
//       </header>

//       {/* Tabs + Logout */}
//       <div className="bg-white border-b border-gray-200">
//         <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
//           <nav className="flex items-center justify-between" aria-label="Tabs">
            
//             {/* Left: Tabs */}
//             <div className="flex space-x-8">
//               {tabs.map((tab) => {
//                 const isActive = location.pathname === tab.path
//                 const Icon = tab.icon
//                 return (
//                   <Link
//                     key={tab.name}
//                     to={tab.path}
//                     className={`
//                       flex items-center gap-2 py-4 px-1 border-b-2 font-medium text-sm
//                       transition-colors duration-200
//                       ${
//                         isActive
//                           ? 'border-primary-500 text-primary-600'
//                           : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
//                       }
//                     `}
//                   >
//                     <Icon className="w-5 h-5" />
//                     {tab.name}
//                   </Link>
//                 )
//               })}
//             </div>

//             {/* Right: Logout Button */}
//             <button
//               onClick={() =>
//                 logout({
//                   logoutParams: {
//                     returnTo: window.location.origin,
//                   },
//                 })
//               }
//               className="
//                 flex items-center gap-2 py-2 px-4
//                 text-sm font-medium text-gray-600
//                 hover:text-red-600 hover:bg-gray-100
//                 rounded-md transition-colors duration-200
//               "
//             >
//               <LogOut className="w-5 h-5" />
//               Logout
//             </button>

//           </nav>
//         </div>
//       </div>

//       {/* Main Content */}
//       <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
//         {children}
//       </main>
//     </div>
//   )
// }

// export default Layout
