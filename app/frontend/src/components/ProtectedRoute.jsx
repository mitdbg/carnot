import { withAuthenticationRequired } from '@auth0/auth0-react';
import { Loader2 } from 'lucide-react';

const ProtectedRoute = ({ component, ...args }) => {
  const Component = withAuthenticationRequired(component, {
    onRedirecting: () => (
      <div className="flex items-center justify-center h-screen">
        <Loader2 className="w-8 h-8 animate-spin text-primary-500" />
      </div>
    ),
  });

  return <Component {...args} />;
};

export default ProtectedRoute;
