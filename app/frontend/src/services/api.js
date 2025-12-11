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
  upload: (file) => {
    const formData = new FormData()
    formData.append('file', file)
    return api.post('/files/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
  },
  listUploaded: () => api.get('/files/upload'),
}

// Datasets API
export const datasetsApi = {
  list: () => api.get('/datasets/'),
  create: (data) => api.post('/datasets/', data),
  get: (id) => api.get(`/datasets/${id}`),
  update: (id, data) => api.put(`/datasets/${id}`, data),
  delete: (id) => api.delete(`/datasets/${id}`),
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
