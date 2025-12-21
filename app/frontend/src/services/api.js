import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api'

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

// Files API
export const filesApi = {
  browse: (path = '') => api.get('/files/browse', { params: { path } }),
  upload: (file, token) => {
    const formData = new FormData()
    formData.append('file', file)
    return api.post('/files/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
        Authorization: `Bearer ${token}`,
      },
    })
  },
  listUploaded: (token) => api.get('/files/upload', {
      headers: { Authorization: `Bearer ${token}` }
    }),
  delete: (filePaths) => api.post('/files/delete', { files: filePaths }),
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

export default api
