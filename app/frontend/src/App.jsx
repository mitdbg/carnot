import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import ProtectedRoute from './components/ProtectedRoute'
import DataManagementPage from './pages/DataManagementPage'
import UserChatPage from './pages/UserChatPage'
import Settings from './pages/Settings'


function App() {
  // const { isAuthenticated, isLoading, error, loginWithRedirect } = useAuth0()

  // // automatically redirect unauthenticated users to Auth0
  // useEffect(() => {
  //   if (!isLoading && !isAuthenticated) {
  //     loginWithRedirect();
  //   }
  // }, [isLoading, isAuthenticated, loginWithRedirect])

  // if (isLoading) return <div className="p-8 text-center text-lg text-gray-600">Loading…</div>
  // if (error) return <div className="p-8 text-center text-lg text-red-600">Error: {error.message}</div>

  // // while redirecting, or if not authenticated yet, render a tiny message
  // if (!isAuthenticated) {
  //   return <div className="p-8 text-center text-lg text-indigo-600">Redirecting to login...</div>
  // }

  // authenticated layout
  return (
    <Layout>
      <Routes>
        <Route 
          path="/"
          element={<ProtectedRoute component={DataManagementPage} />}
        />
        <Route
          path="/chat"
          element={<ProtectedRoute component={UserChatPage} />}
        />
        <Route
          path="/settings"
          element={<ProtectedRoute component={Settings} />}
        />
      </Routes>
    </Layout>
  )
}

export default App;
