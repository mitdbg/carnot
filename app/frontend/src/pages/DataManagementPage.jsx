import path from 'path';
import { useState, useEffect } from 'react'
import { Plus, Loader2, Database, Calendar, FileText, Trash2, X, Eye, Save, ArrowLeft } from 'lucide-react'
import FileBrowser from '../components/DataManagement/FileBrowser'
import SearchChatbot from '../components/DataManagement/SearchChatbot'
import DatasetAnnotation from '../components/DataManagement/DatasetAnnotation'
import { datasetsApi } from '../services/api'
import { useApiToken } from '../hooks/useApiToken';

function DataManagementPage() {
  const getValidToken = useApiToken();
  const [datasets, setDatasets] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [success, setSuccess] = useState(null)
  const [selectedDataset, setSelectedDataset] = useState(null)
  const [viewingDataset, setViewingDataset] = useState(false)
  const [selectedFiles, setSelectedFiles] = useState(new Set())
  const [isCreatingMode, setIsCreatingMode] = useState(false)
  const [datasetName, setDatasetName] = useState('')
  const [annotation, setAnnotation] = useState('')
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    try {
      setLoading(true)
      setError(null)
      const token = await getValidToken();
      if (!token) return;
      const datasetsRes = await datasetsApi.list(token)
      setDatasets(datasetsRes.data)
    } catch (err) {
      setError('Failed to load data: ' + err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleFileToggle = (filePath, bulkSelection = null) => {
    if (bulkSelection) {
      // Handles 'Select All' and 'Directory Toggles'
      setSelectedFiles(bulkSelection)
    } else {
      // Handles individual file selection
      const newSelected = new Set(selectedFiles)
      if (newSelected.has(filePath)) {
        newSelected.delete(filePath)
      } else {
        newSelected.add(filePath)
      }
      setSelectedFiles(newSelected)
    }
  }

  const handleChatbotSelect = (files) => {
    const newSelected = new Set(selectedFiles)
    files.forEach((file) => newSelected.add(file.file_path))
    setSelectedFiles(newSelected)
  }

  const handleSaveDataset = async () => {
    if (!datasetName.trim() || !annotation.trim() || selectedFiles.size === 0) {
      setError('Please provide a name, description, and select at least one file.')
      return
    }

    try {
      setSaving(true)
      const token = await getValidToken();
      if (!token) return;

      await datasetsApi.create({
        name: datasetName,
        shared: false,
        annotation: annotation,
        files: Array.from(selectedFiles),
      }, token)

      setSuccess('Dataset created successfully!')
      resetCreationState()
      loadData()
    } catch (err) {
      setError('Failed to create dataset: ' + err.message)
    } finally {
      setSaving(false)
    }
  }

  const resetCreationState = () => {
    setIsCreatingMode(false)
    setDatasetName('')
    setAnnotation('')
    setSelectedFiles(new Set())
  }

  const handleDeleteDataset = async (id) => {
    if (!window.confirm('Are you sure you want to delete this dataset?')) {
      return
    }

    try {
      setError(null)
      const token = await getValidToken();
      if (!token) return;
      await datasetsApi.delete(id, token)
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
      const token = await getValidToken();
      if (!token) return;
      const response = await datasetsApi.get(datasetId, token)
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
      {/* Dynamic Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          {isCreatingMode && (
            <button 
              onClick={resetCreationState} 
              className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              title="Go back"
            >
              <ArrowLeft className="w-6 h-6" />
            </button>
          )}
          <div>
            <h1 className="text-3xl font-bold text-gray-900">
              {isCreatingMode ? 'Create New Dataset' : 'Data Management'}
            </h1>
            <p className="mt-1 text-gray-600">
              {isCreatingMode ? 'Configure your dataset details' : 'Manage your datasets and browse files'}
            </p>
          </div>
        </div>
        
        <div className="flex items-center gap-3">
          {!isCreatingMode ? (
            <button
              onClick={() => setIsCreatingMode(true)}
              className="flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 shadow-md transition-all"
            >
              <Plus className="w-5 h-5" />
              New Dataset
            </button>
          ) : (
            <>
              {/* NEW CANCEL BUTTON */}
              <button
                onClick={resetCreationState}
                disabled={saving}
                className="px-6 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors disabled:opacity-50"
              >
                Cancel
              </button>

              {/* SAVE BUTTON */}
              <button
                onClick={handleSaveDataset}
                disabled={saving}
                className="flex items-center gap-2 px-6 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 shadow-md transition-all disabled:opacity-50"
              >
                {saving ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Saving...
                  </>
                ) : (
                  <>
                    <Save className="w-5 h-5" />
                    Save Dataset
                  </>
                )}
              </button>
            </>
          )}
        </div>
      </div>

      {/* Notifications */}
      {error && <div className="bg-red-50 border border-red-200 text-red-800 px-4 py-3 rounded-lg">{error}</div>}
      {success && <div className="bg-green-50 border border-green-200 text-green-800 px-4 py-3 rounded-lg">{success}</div>}

      {/* Creation HUD */}
      {isCreatingMode && (
        <div className="bg-primary-50 border border-primary-200 px-4 py-3 rounded-lg flex justify-between items-center">
          <p className="text-primary-800 font-medium">
            Select files below and add annotations. 
            <span className="ml-4 bg-primary-200 px-2 py-1 rounded text-sm">{selectedFiles.size} files selected</span>
          </p>
        </div>
      )}

      {/* Main Grid: File Browser + Optional Chatbot */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className={isCreatingMode ? "lg:col-span-2" : "lg:col-span-3"}>
          <FileBrowser selectedFiles={selectedFiles} onFileToggle={handleFileToggle} />
        </div>
        {isCreatingMode && (
          <div>
            <SearchChatbot onSelectFiles={handleChatbotSelect} />
          </div>
        )}
      </div>

      {/* Creation Form or Dataset List */}
      {isCreatingMode ? (
        <DatasetAnnotation
          datasetName={datasetName}
          annotation={annotation}
          onNameChange={setDatasetName}
          onAnnotationChange={setAnnotation}
        />
      ) : (
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
      )}

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

