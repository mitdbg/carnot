import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api'
const MAX_FILES_PER_REQUEST = 50; // cap for file browsing to prevent overload

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Config API
export const configApi = {
  getConfig: () => api.get('/config/'),
}

// Settings API
export const settingsApi = {
  getSettings: (token) =>
    api.get('/settings/', {
      headers: { Authorization: `Bearer ${token}` }
    }),
  updateApiKeys: (keys, token) =>
    api.post('/settings/keys', keys, {
      headers: { Authorization: `Bearer ${token}` }
    }),
  uploadEnvFile: (file, token) => {
    const formData = new FormData();
    formData.append('file', file);
    return api.post('/settings/env', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
        Authorization: `Bearer ${token}`,
      },
    });
  },
}

// Conversations API
export const conversationsApi = {
  list: (token) =>
    api.get('/conversations/', {
      headers: { Authorization: `Bearer ${token}` }
    }),
  get: (id, token) =>
    api.get(`/conversations/${id}`, {
      headers: { Authorization: `Bearer ${token}` }
    }),
  delete: (id, token) =>
    api.delete(`/conversations/${id}`, {
      headers: { Authorization: `Bearer ${token}` }
    }),
}

// Workspaces API
export const workspacesApi = {
  list: (token) =>
    api.get('/workspaces/', {
      headers: { Authorization: `Bearer ${token}` }
    }),
  get: (id, token) =>
    api.get(`/workspaces/${id}`, {
      headers: { Authorization: `Bearer ${token}` }
    }),
  create: (data, token) =>
    api.post('/workspaces/', data, {
      headers: { Authorization: `Bearer ${token}` }
    }),
  update: (id, data, token) =>
    api.put(`/workspaces/${id}`, data, {
      headers: { Authorization: `Bearer ${token}` }
    }),
  delete: (id, token) =>
    api.delete(`/workspaces/${id}`, {
      headers: { Authorization: `Bearer ${token}` }
    }),
}

// Files API
export const filesApi = {
  browse: (token, path = '', limit = MAX_FILES_PER_REQUEST, continuationToken = null) => {
    const params = { path, limit };
    if (continuationToken) {
      params.continuation_token = continuationToken;
    }
    return api.get('/files/browse', {
      params,
      headers: { Authorization: `Bearer ${token}` }
    });
  },
  createDirectory: (path, name, token) => {
    return api.post('/files/create-directory', { path, name }, {
      headers: { Authorization: `Bearer ${token}` }
    });
  },
  upload: (file, token, destinationPath = '') => {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('path', destinationPath);
    return api.post('/files/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
        Authorization: `Bearer ${token}`,
      },
    })
  },
  delete: (filePaths) => api.post('/files/delete', { files: filePaths }),
  expandPaths: (paths, token) => 
    api.post('/files/expand-paths', paths, {
      headers: { Authorization: `Bearer ${token}` }
    }),
}

// Datasets API
export const datasetsApi = {
  list: (token) =>
    api.get('/datasets/', {
      headers: { Authorization: `Bearer ${token}` }
    }),
  create: (data, token) =>
    api.post('/datasets/', data, {
      headers: { Authorization: `Bearer ${token}` }
    }),
  get: (id, token) =>
    api.get(`/datasets/${id}`, {
      headers: { Authorization: `Bearer ${token}` }
    }),
  update: (id, data, token) =>
    api.put(`/datasets/${id}`, data, {
      headers: { Authorization: `Bearer ${token}` }
    }),
  delete: (id, token) =>
    api.delete(`/datasets/${id}`, {
      headers: { Authorization: `Bearer ${token}` }
    }),
}

// Search API
export const searchApi = {
  search: (query, paths = null, token) => 
    api.post('/search',
      { query, paths },
      {headers: { Authorization: `Bearer ${token}` }}
    ),
}

// Notebook API
export const notebookApi = {
  create: (data, token) =>
    api.post('/query/execute-jupyter', data, {
      headers: { Authorization: `Bearer ${token}` }
    }),
  get: (notebookId, token) =>
    api.get(`/query/notebook/${notebookId}`, {
      headers: { Authorization: `Bearer ${token}` }
    }),
  close: (notebookId, token) =>
    api.delete(`/query/notebook/${notebookId}`, {
      headers: { Authorization: `Bearer ${token}` }
    }),
  addCell: (notebookId, data, token) =>
    api.post(`/query/notebook/${notebookId}/cells`, data, {
      headers: { Authorization: `Bearer ${token}` }
    }),
  deleteCell: (notebookId, cellId, token) =>
    api.delete(`/query/notebook/${notebookId}/cells/${cellId}`, {
      headers: { Authorization: `Bearer ${token}` }
    }),
  moveCell: (notebookId, data, token) =>
    api.post(`/query/notebook/${notebookId}/cells/move`, data, {
      headers: { Authorization: `Bearer ${token}` }
    }),
}

export default api
