import { FileText } from 'lucide-react'

function DatasetAnnotation({ datasetName, annotation, onNameChange, onAnnotationChange }) {
  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center gap-2">
        <FileText className="w-5 h-5" />
        Dataset Information
      </h2>
      
      <div className="space-y-4">
        {/* Dataset Name */}
        <div>
          <label htmlFor="dataset-name" className="block text-sm font-medium text-gray-700 mb-2">
            Dataset Name <span className="text-red-500">*</span>
          </label>
          <input
            type="text"
            id="dataset-name"
            value={datasetName}
            onChange={(e) => onNameChange(e.target.value)}
            placeholder="Enter a unique name for this dataset"
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          />
        </div>

        {/* Annotation */}
        <div>
          <label htmlFor="annotation" className="block text-sm font-medium text-gray-700 mb-2">
            Annotation / Description <span className="text-red-500">*</span>
          </label>
          <textarea
            id="annotation"
            value={annotation}
            onChange={(e) => onAnnotationChange(e.target.value)}
            placeholder="Describe this dataset - what it contains, its purpose, any relevant metadata..."
            rows={4}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-none"
          />
          <p className="mt-2 text-sm text-gray-500">
            Add any metadata or information that will help you identify and use this dataset later.
          </p>
        </div>
      </div>
    </div>
  )
}

export default DatasetAnnotation

