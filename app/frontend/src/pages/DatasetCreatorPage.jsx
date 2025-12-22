import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowLeft, Save, Loader2 } from 'lucide-react'
import FileBrowser from '../components/DatasetCreator/FileBrowser'
import SearchChatbot from '../components/DatasetCreator/SearchChatbot'
import DatasetAnnotation from '../components/DatasetCreator/DatasetAnnotation'
import { datasetsApi } from '../services/api'
import { useApiToken } from '../hooks/useApiToken';

function DatasetCreatorPage() {
  const getValidToken = useApiToken();
  const navigate = useNavigate()
  const [selectedFiles, setSelectedFiles] = useState(new Set())
  const [datasetName, setDatasetName] = useState('')
  const [annotation, setAnnotation] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleFileToggle = (filePath, newSet = null) => {
    if (newSet !== null) {
      // Allow passing a new set directly (for bulk operations)
      setSelectedFiles(newSet)
      return
    }
    
    const newSelected = new Set(selectedFiles)
    if (newSelected.has(filePath)) {
      newSelected.delete(filePath)
    } else {
      newSelected.add(filePath)
    }
    setSelectedFiles(newSelected)
  }

  const handleChatbotSelect = (files) => {
    const newSelected = new Set(selectedFiles)
    files.forEach((file) => {
      newSelected.add(file.file_path)
    })
    setSelectedFiles(newSelected)
  }

  const handleSaveDataset = async () => {
    // Validation
    if (!datasetName.trim()) {
      setError('Please enter a dataset name')
      return
    }
    if (!annotation.trim()) {
      setError('Please enter an annotation')
      return
    }
    if (selectedFiles.size === 0) {
      setError('Please select at least one file')
      return
    }

    try {
      setLoading(true)
      setError(null)
      const token = await getValidToken();
      if (!token) return;

      // Convert selected files to Array
      const files = Array.from(selectedFiles)
      await datasetsApi.create({
        name: datasetName,
        shared: false, // TODO: allow shared datasets
        annotation: annotation,
        files: files,
      }, token)

      // Navigate back to data management page
      navigate('/')
    } catch (err) {
      setError('Failed to create dataset: ' + (err.response?.data?.detail || err.message))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <button
            onClick={() => navigate('/')}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <ArrowLeft className="w-6 h-6" />
          </button>
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Create Dataset</h1>
            <p className="mt-1 text-gray-600">
              Select files and add annotations
            </p>
          </div>
        </div>
        <button
          onClick={handleSaveDataset}
          disabled={loading}
          className="flex items-center gap-2 px-6 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors duration-200 shadow-md hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? (
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
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 text-red-800 px-4 py-3 rounded-lg">
          {error}
        </div>
      )}

      {/* Selected Files Counter */}
      <div className="bg-primary-50 border border-primary-200 px-4 py-3 rounded-lg">
        <p className="text-primary-800">
          <span className="font-semibold">{selectedFiles.size}</span> file(s) selected
        </p>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* File Browser - Takes 2 columns on large screens */}
        <div className="lg:col-span-2">
          <FileBrowser
            selectedFiles={selectedFiles}
            onFileToggle={handleFileToggle}
          />
        </div>

        {/* Chatbot - Takes 1 column */}
        <div>
          <SearchChatbot onSelectFiles={handleChatbotSelect} />
        </div>
      </div>

      {/* Annotation Form */}
      <DatasetAnnotation
        datasetName={datasetName}
        annotation={annotation}
        onNameChange={setDatasetName}
        onAnnotationChange={setAnnotation}
      />
    </div>
  )
}

export default DatasetCreatorPage

