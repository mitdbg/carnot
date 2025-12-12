import path from 'path';
import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Plus, Upload, Loader2, Database, Calendar, FileText, Trash2, X, Eye } from 'lucide-react'
import { datasetsApi, filesApi } from '../services/api'

function DataManagementPage() {
  const navigate = useNavigate()
  const [datasets, setDatasets] = useState([])
  const [uploadedFiles, setUploadedFiles] = useState([])
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState(null)
  const [success, setSuccess] = useState(null)
  const [selectedDataset, setSelectedDataset] = useState(null)
  const [viewingDataset, setViewingDataset] = useState(false)

  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    try {
      setLoading(true)
      setError(null)
      const [datasetsRes, filesRes] = await Promise.all([
        datasetsApi.list(),
        filesApi.listUploaded(),
      ])
      setDatasets(datasetsRes.data)
      setUploadedFiles(filesRes.data)
    } catch (err) {
      setError('Failed to load data: ' + err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleFileUpload = async (e) => {
    const file = e.target.files[0]
    if (!file) return

    try {
      setUploading(true)
      setError(null)
      await filesApi.upload(file)
      setSuccess('File uploaded successfully!')
      setTimeout(() => setSuccess(null), 3000)
      loadData()
    } catch (err) {
      setError('Failed to upload file: ' + err.message)
    } finally {
      setUploading(false)
    }
  }

  const handleDeleteDataset = async (id) => {
    if (!window.confirm('Are you sure you want to delete this dataset?')) {
      return
    }

    try {
      setError(null)
      await datasetsApi.delete(id)
      setSuccess('Dataset deleted successfully!')
      setTimeout(() => setSuccess(null), 3000)
      loadData()
    } catch (err) {
      setError('Failed to delete dataset: ' + err.message)
    }
  }

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    })
  }

  const handleViewDataset = async (datasetId) => {
    try {
      setError(null)
      const response = await datasetsApi.get(datasetId)
      setSelectedDataset(response.data)
      setViewingDataset(true)
    } catch (err) {
      setError('Failed to load dataset details: ' + err.message)
    }
  }

  const closeModal = () => {
    setViewingDataset(false)
    setSelectedDataset(null)
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 animate-spin text-primary-500" />
      </div>
    )
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Data Management</h1>
          <p className="mt-2 text-gray-600">
            Manage your datasets and upload files
          </p>
        </div>
        <button
          onClick={() => navigate('/datasets/create')}
          className="flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors duration-200 shadow-md hover:shadow-lg"
        >
          <Plus className="w-5 h-5" />
          Create Dataset
        </button>
      </div>

      {/* Notifications */}
      {error && (
        <div className="bg-red-50 border border-red-200 text-red-800 px-4 py-3 rounded-lg">
          {error}
        </div>
      )}
      {success && (
        <div className="bg-green-50 border border-green-200 text-green-800 px-4 py-3 rounded-lg">
          {success}
        </div>
      )}

      {/* File Upload Section */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center gap-2">
          <Upload className="w-5 h-5" />
          Upload Files
        </h2>
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-primary-400 transition-colors">
          <input
            type="file"
            id="file-upload"
            className="hidden"
            onChange={handleFileUpload}
            disabled={uploading}
          />
          <label
            htmlFor="file-upload"
            className="cursor-pointer flex flex-col items-center"
          >
            {uploading ? (
              <Loader2 className="w-12 h-12 text-gray-400 animate-spin mb-3" />
            ) : (
              <Upload className="w-12 h-12 text-gray-400 mb-3" />
            )}
            <span className="text-lg font-medium text-gray-700">
              {uploading ? 'Uploading...' : 'Click to upload a file'}
            </span>
            <span className="text-sm text-gray-500 mt-1">
              or drag and drop
            </span>
          </label>
        </div>

        {/* Uploaded Files List */}
        {uploadedFiles.length > 0 && (
          <div className="mt-6">
            <h3 className="text-sm font-medium text-gray-700 mb-3">
              Recently Uploaded ({uploadedFiles.length})
            </h3>
            <div className="space-y-2 max-h-40 overflow-y-auto">
              {uploadedFiles.slice(0, 10).map((file) => (
                <div
                  key={file.id}
                  className="flex items-center gap-2 text-sm text-gray-600 bg-gray-50 px-3 py-2 rounded"
                >
                  <FileText className="w-4 h-4" />
                  <span className="flex-1">{path.basename(file.file_path)}</span>
                  <span className="text-xs text-gray-400">
                    {formatDate(file.upload_date)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Datasets List */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center gap-2">
          <Database className="w-5 h-5" />
          Datasets ({datasets.length})
        </h2>
        
        {datasets.length === 0 ? (
          <div className="text-center py-12 text-gray-500">
            <Database className="w-16 h-16 mx-auto mb-4 text-gray-300" />
            <p className="text-lg">No datasets yet</p>
            <p className="text-sm mt-2">Create your first dataset to get started</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {datasets.map((dataset) => (
              <div
                key={dataset.id}
                className="border border-gray-200 rounded-lg p-4 hover:shadow-lg transition-shadow duration-200 cursor-pointer"
                onClick={() => handleViewDataset(dataset.id)}
              >
                <div className="flex items-start justify-between mb-3">
                  <h3 className="text-lg font-semibold text-gray-800 flex-1">
                    {dataset.name}
                  </h3>
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      handleDeleteDataset(dataset.id)
                    }}
                    className="text-red-500 hover:text-red-700 p-1"
                    title="Delete dataset"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
                
                <p className="text-sm text-gray-600 mb-4 line-clamp-2">
                  {dataset.annotation}
                </p>
                
                <div className="flex items-center justify-between text-sm text-gray-500">
                  <div className="flex items-center gap-1">
                    <FileText className="w-4 h-4" />
                    <span>{dataset.file_count} files</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <Calendar className="w-4 h-4" />
                    <span>{formatDate(dataset.created_at)}</span>
                  </div>
                </div>
                
                <div className="mt-3 pt-3 border-t border-gray-200 text-xs text-gray-500 flex items-center gap-1">
                  <Eye className="w-3 h-3" />
                  <span>Click to view files</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Dataset Details Modal */}
      {viewingDataset && selectedDataset && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg shadow-xl max-w-3xl w-full max-h-[80vh] overflow-hidden">
            {/* Modal Header */}
            <div className="flex items-center justify-between p-6 border-b border-gray-200">
              <div className="flex-1">
                <h2 className="text-2xl font-bold text-gray-900">{selectedDataset.name}</h2>
                <p className="text-sm text-gray-500 mt-1">
                  {selectedDataset.files.length} files • Created {formatDate(selectedDataset.created_at)}
                </p>
              </div>
              <button
                onClick={closeModal}
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <X className="w-6 h-6" />
              </button>
            </div>

            {/* Modal Body */}
            <div className="p-6 overflow-y-auto max-h-[60vh]">
              {/* Annotation */}
              <div className="mb-6">
                <h3 className="text-sm font-semibold text-gray-700 mb-2">Description</h3>
                <p className="text-gray-600">{selectedDataset.annotation}</p>
              </div>

              {/* Files List */}
              <div>
                <h3 className="text-sm font-semibold text-gray-700 mb-3">
                  Files ({selectedDataset.files.length})
                </h3>
                <div className="space-y-2">
                  {selectedDataset.files.map((file) => (
                    <div
                      key={file.id}
                      className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
                    >
                      <FileText className="w-5 h-5 text-gray-400 flex-shrink-0" />
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-gray-900 truncate">
                          {path.basename(file.file_path)}
                        </p>
                        <p className="text-xs text-gray-500 truncate">
                          {file.file_path}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Modal Footer */}
            <div className="p-6 border-t border-gray-200 bg-gray-50">
              <button
                onClick={closeModal}
                className="w-full px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default DataManagementPage

