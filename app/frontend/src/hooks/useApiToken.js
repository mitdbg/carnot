import { useAuth0 } from '@auth0/auth0-react';

export const useApiToken = () => {
  const { getAccessTokenSilently, loginWithRedirect } = useAuth0();

  const getValidToken = async () => {
    try {
      return await getAccessTokenSilently();
    } catch (err) {
      const isAuthError = [
        'login_required',
        'consent_required',
        'missing_refresh_token'
      ].includes(err.error);

      if (isAuthError) {
        console.warn("Auth session invalid. Redirecting to login...");
        loginWithRedirect();
        return null; // Return null so the component knows to stop
      }
      throw err;
    }
  };

  return getValidToken;
};
