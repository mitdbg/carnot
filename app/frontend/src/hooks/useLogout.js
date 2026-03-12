import { useAuth0 } from '@auth0/auth0-react'

const isLocalAuthBypass = import.meta.env.VITE_LOCAL_AUTH_BYPASS === 'true'

const useLocalLogout = () => {
  return () => {
    localStorage.clear()
    sessionStorage.clear()
  }
}

const useAuth0Logout = () => {
  const { logout } = useAuth0()
  return () => {
    localStorage.clear()
    sessionStorage.clear()
    logout({ logoutParams: { returnTo: window.location.origin } })
  }
}

export const useLogout = isLocalAuthBypass ? useLocalLogout : useAuth0Logout
