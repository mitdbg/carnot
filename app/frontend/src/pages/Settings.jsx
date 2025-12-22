import { useState, useEffect } from 'react';
import { settingsApi } from '../services/api';
import { useApiToken } from '../hooks/useApiToken';

const Settings = () => {
  const getValidToken = useApiToken();

  // state
  const [keys, setKeys] = useState({
    OPENAI_API_KEY: '',
    ANTHROPIC_API_KEY: '',
    GEMINI_API_KEY: '',
    TOGETHER_API_KEY: '',
  });
  const [selectedFile, setSelectedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState({ type: '', text: '' });

  // fetch existing settings on load
  useEffect(() => {
    const fetchSettings = async () => {
      try {
        // use the token to authenticate
        const token = await getValidToken();
        if (!token) return;

        // retrieve user-specific settings from the backend
        const response = await settingsApi.getSettings(token); 
        if(response.data) setKeys(prev => ({ ...prev, ...response.data }));
      } catch (error) {
        console.error("Could not fetch settings", error);
      }
    };

    fetchSettings();
  }, []);

  // Handle manual input changes
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setKeys((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  // Handle file selection
  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  // Submit Manual Keys
  const handleSaveKeys = async (e) => {
    e.preventDefault();
    setLoading(true);
    setMessage({ type: '', text: '' });

    try {
      const token = await getValidToken();
      if (!token) return;
      await settingsApi.updateApiKeys(keys, token);
      setMessage({ type: 'success', text: 'API Keys saved successfully!' });
      setKeys({
        OPENAI_API_KEY: '',
        ANTHROPIC_API_KEY: '',
        GEMINI_API_KEY: '',
        TOGETHER_API_KEY: '',
      });
    } catch (error) {
      console.error(error);
      setMessage({ type: 'error', text: 'Failed to save API Keys.' });
    } finally {
      setLoading(false);
    }
  };

  // Submit .env File
  const handleUploadEnv = async (e) => {
    e.preventDefault();
    if (!selectedFile) {
      setMessage({ type: 'error', text: 'Please select a .env file first.' });
      return;
    }

    setLoading(true);
    setMessage({ type: '', text: '' });

    try {
      const token = await getValidToken();
      if (!token) return;
      await settingsApi.uploadEnvFile(selectedFile, token);
      setMessage({ type: 'success', text: '.env file uploaded and parsed successfully.' });
      setSelectedFile(null);
      // Reset file input visually
      document.getElementById('env-file-input').value = ''; 
    } catch (error) {
      console.error(error);
      setMessage({ type: 'error', text: 'Failed to upload .env file.' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto p-6">
      <h1 className="text-2xl font-bold mb-6">Settings</h1>

      {/* Feedback Message */}
      {message.text && (
        <div className={`p-4 mb-6 rounded ${message.type === 'error' ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'}`}>
          {message.text}
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">

        {/* Section 1: Manual Key Entry */}
        <div className="bg-white p-6 rounded shadow border">
          <h2 className="text-xl font-semibold mb-4">API Key Configuration</h2>
          <p className="text-gray-600 mb-4 text-sm">Enter your LLM provider API keys manually.</p>

          <form onSubmit={handleSaveKeys}>
            {Object.keys(keys).map((keyName) => (
              <div key={keyName} className="mb-4">
                <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor={keyName}>
                  {keyName.replace(/_/g, ' ')}
                </label>
                <input
                  type="password"
                  name={keyName}
                  id={keyName}
                  value={keys[keyName]}
                  onChange={handleInputChange}
                  placeholder={`...`}
                  className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                />
              </div>
            ))}
            <button
              type="submit"
              disabled={loading}
              className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline w-full"
            >
              {loading ? 'Saving...' : 'Save Keys'}
            </button>
          </form>
        </div>

        {/* Section 2: .env Upload */}
        <div className="bg-white p-6 rounded shadow border h-fit">
          <h2 className="text-xl font-semibold mb-4">Environment File Upload</h2>
          <p className="text-gray-600 mb-4 text-sm">Alternatively, upload a local <code>.env</code> file containing your configuration.</p>

          <form onSubmit={handleUploadEnv}>
            <div className="mb-6">
              <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="env-file-input">
                Select File
              </label>
              <input
                type="file"
                id="env-file-input"
                accept=".env"
                onChange={handleFileChange}
                className="block w-full text-sm text-gray-500
                  file:mr-4 file:py-2 file:px-4
                  file:rounded-full file:border-0
                  file:text-sm file:font-semibold
                  file:bg-blue-50 file:text-blue-700
                  hover:file:bg-blue-100"
              />
            </div>
            <button
              type="submit"
              disabled={loading || !selectedFile}
              className={`font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline w-full ${
                !selectedFile ? 'bg-gray-300 cursor-not-allowed' : 'bg-green-500 hover:bg-green-700 text-white'
              }`}
            >
              {loading ? 'Uploading...' : 'Upload .env'}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};

export default Settings;
