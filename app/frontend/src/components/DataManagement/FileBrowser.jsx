import { useState, useEffect } from 'react'
import { 
  ChevronRight, 
  ChevronDown, 
  Folder, 
  FolderPlus, 
  File, 
  Loader2, 
  Home, 
  Check, 
  CheckSquare, 
  Square, 
  Trash2, 
  Upload as UploadIcon,
  X as Close
} from 'lucide-react'
import { configApi, filesApi } from '../../services/api'
import { useApiToken } from '../../hooks/useApiToken';

function formatFileSize(bytes) {
  if (bytes === 0) return '0 Bytes'
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i]
}

function FileBrowser({ selectedFiles, onFileToggle }) {
  const getValidToken = useApiToken()
  const [currentPath, setCurrentPath] = useState('')
  const [items, setItems] = useState([])
  const [baseDirPathFull, setBaseDirPathFull] = useState('')
  const [dataDirPathFull, setDataDirPathFull] = useState('')
  const [sharedDataDirPathFull, setSharedDataDirPathFull] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [expandedDirs, setExpandedDirs] = useState(new Set())
  const [directorySelectionState, setDirectorySelectionState] = useState(new Map())
  const [isCreatingDirectory, setIsCreatingDirectory] = useState(false)
  const [newDirectoryName, setNewDirectoryName] = useState('')
  const [isCreating, setIsCreating] = useState(false)
  const [isDragging, setIsDragging] = useState(false)

  useEffect(() => {
    const fetchConfig = async () => {
        try {
            const configRes = await configApi.getConfig();
            setBaseDirPathFull(configRes.data.base_dir);
            setDataDirPathFull(configRes.data.data_dir);
            setSharedDataDirPathFull(configRes.data.shared_data_dir);
            loadDirectory('');
        } catch (err) {
            setError('Failed to load config: ' + err.message)
        }
    }
    fetchConfig();
  }, []);

  useEffect(() => {
    loadDirectory(currentPath)
  }, [currentPath])

  // Drag Handlers
  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => setIsDragging(false);

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    handleFileUpload(e.dataTransfer.files);
  };

  const loadDirectory = async (path) => {
    try {
      setLoading(true)
      setError(null)
      const token = await getValidToken();
      if (!token) return;

      const response = await filesApi.browse(token, path)
      const loadedItems = response.data || []
      setItems(loadedItems)
    } catch (err) {
      setError('Failed to load directory: ' + err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleFileUpload = async (files, event = null) => {
    if (!files || files.length === 0) return;
    
    try {
      setLoading(true);
      setError(null);
      const token = await getValidToken();
      if (!token) return;

      // support multiple files if dragged/selected at once
      const uploadPromises = Array.from(files).map(file => 
        filesApi.upload(file, token, currentPath)
      );
      await Promise.all(uploadPromises);

      // refresh the current directory to show the new files
      loadDirectory(currentPath);

      // reset the event value to allow re-uploading the same file if needed
      if (event && event.target) {
        event.target.value = '';
      }
    } catch (err) {
      setError('Upload failed: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateDirectory = async () => {
    if (!newDirectoryName.trim()) return;

    try {
      setIsCreating(true);
      const token = await getValidToken();
      if (!token) return;

      // API call to create the directory
      await filesApi.createDirectory(currentPath, newDirectoryName.trim(), token);

      // Reset state
      setNewDirectoryName('');
      setIsCreatingDirectory(false);

      // Refresh list to show the new folder
      loadDirectory(currentPath);
    } catch (err) {
      setError('Failed to create folder: ' + (err.response?.data?.detail || err.message));
    } finally {
      setIsCreating(false);
    }
  };

  const handleItemClick = (item) => {
    if (item.is_directory) {
      setCurrentPath(item.path)
      setExpandedDirs(new Set([...expandedDirs, item.path]))
    }
  }

  const handleCheckboxChange = (item, event) => {
    event.stopPropagation() // Prevent navigation when clicking checkbox

    if (item.is_directory) {
      handleDirectoryToggle(item)
    } else {
      onFileToggle(item.path)
    }
  }

  const isItemSelected = (item) => {
    return selectedFiles.has(item.path);
  };

  const checkDirectorySelection = async (directoryPath) => {
    // If the directory path itself is in the selection set, it's selected
    if (selectedFiles.has(directoryPath)) return true;

    try {
      const allFiles = await getAllFilesInDirectory(directoryPath);
      if (allFiles.length === 0) return false;
      // Check if all files within are selected (legacy behavior support)
      return allFiles.every(filePath => selectedFiles.has(filePath));
    } catch (err) {
      return false;
    }
  };

  // Update directory selection states when selectedFiles or items change
  useEffect(() => {
    const updateDirectoryStates = async () => {
      const newState = new Map()
      for (const item of items) {
        if (item.is_directory) {
          try {
            const allFiles = await getAllFilesInDirectory(item.path)
            if (allFiles.length > 0) {
              const allSelected = allFiles.every(filePath => {
                return selectedFiles.has(filePath)
              })
              newState.set(item.path, allSelected)
            } else {
              newState.set(item.path, false)
            }
          } catch (err) {
            newState.set(item.path, false)
          }
        }
      }
      setDirectorySelectionState(newState)
    }
    if (items.length > 0) {
      updateDirectoryStates()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedFiles, items])

  // state to manage Select/Delete button visibility
  const hasSelectedFiles = selectedFiles.size > 0;

  // handle the deletion of selected files and browser refresh
  const handleDeleteSelected = async () => {
    if (!window.confirm(`Are you sure you want to permanently delete ${selectedFiles.size} selected file(s)? This action cannot be undone.`)) {
      return
    }

    try {
      setLoading(true)
      setError(null)

      const filesToDelete = Array.from(selectedFiles)

      // perform all deletions in parallel
      await filesApi.delete(filesToDelete)

      // clear the selection in the parent component; we use onFileToggle with null and an empty Set
      // to trigger the bulk update logic defined in DataManagementPage's handleFileToggle.
      onFileToggle(null, new Set())

      // refresh the file list by navigating to the root; this state change (currentPath = '') will
      // automatically trigger the useEffect to call loadDirectory('')
      navigateToRoot()

    } catch (err) {
      setError('Failed to delete files: ' + (err.response?.data?.detail || err.message))
    } finally {
      setLoading(false)
    }
  }

  const getAllFilesInDirectory = async (dirPath) => {
    const allFiles = []
    const token = await getValidToken();
    if (!token) return allFiles;

    const loadDirRecursive = async (path) => {
      try {
        const response = await filesApi.browse(token, path)
        const items = response.data || []

        for (const item of items) {
          if (item.is_directory) {
            // recursively load subdirectory
            await loadDirRecursive(item.path)
          } else {
            // add file path to list of files
            allFiles.push(item.path)
          }
        }
      } catch (err) {
        console.error(`Failed to load directory ${path}:`, err)
      }
    }

    await loadDirRecursive(dirPath)
    return allFiles
  }

  const handleDirectoryToggle = async (directory) => {
    const allFiles = await getAllFilesInDirectory(directory.path);
    const directoryPath = directory.path;
    
    // A directory is considered "active" if its path is selected 
    // OR all its files are selected
    const isCurrentlySelected = selectedFiles.has(directoryPath);

    const newSelected = new Set(selectedFiles);

    if (isCurrentlySelected) {
      // Deselect the directory itself
      newSelected.delete(directoryPath);
      // Deselect all nested files
      allFiles.forEach(filePath => newSelected.delete(filePath));
    } else {
      // Select the directory itself (this allows deleting empty folders!)
      newSelected.add(directoryPath);
      // Select all nested files (for dataset creation compatibility)
      allFiles.forEach(filePath => newSelected.add(filePath));
    }

    onFileToggle(null, newSelected);
  };

  const getCurrentDirectoryFiles = () => {
    return items.filter(item => !item.is_directory)
  }

  const getCurrentDirectoryFolders = () => {
    return items.filter(item => item.is_directory)
  }

  const areAllItemsSelected = async () => {
    const files = getCurrentDirectoryFiles()
    const folders = getCurrentDirectoryFolders()

    if (files.length === 0 && folders.length === 0) return false

    // Check all files are selected
    const allFilesSelected = files.length === 0 || files.every(file => isItemSelected(file))

    // Check all folders are fully selected
    let allFoldersSelected = true
    for (const folder of folders) {
      const folderSelected = await checkDirectorySelection(folder.path)
      if (!folderSelected) {
        allFoldersSelected = false
        break
      }
    }

    return allFilesSelected && allFoldersSelected
  }

  const handleSelectAll = async () => {
    const files = getCurrentDirectoryFiles()
    const folders = getCurrentDirectoryFolders()
    const allSelected = await areAllItemsSelected()

    const newSelected = new Set(selectedFiles)

    if (allSelected) {
      // Deselect all files in current directory
      files.forEach(file => {
        newSelected.delete(file.path)
      })

      // Deselect all folders (and their contents)
      for (const folder of folders) {
        const allFiles = await getAllFilesInDirectory(folder.path)
        allFiles.forEach(filePath => {
          newSelected.delete(filePath)
        })
      }
    } else {
      // Select all files in current directory
      files.forEach(file => {
        newSelected.add(file.path)
      })

      // Select all folders (and their contents)
      for (const folder of folders) {
        const allFiles = await getAllFilesInDirectory(folder.path)
        allFiles.forEach(filePath => {
          newSelected.add(filePath)
        })
      }
    }

    // update all at once
    onFileToggle(null, newSelected)
  }

  const navigateUp = () => {
    if (currentPath) {
        const base = baseDirPathFull.replace(/\/$/, '');
        const dataDir = dataDirPathFull.replace(/\/$/, '');
        const normalizedCurrentPath = currentPath.replace(/\/$/, '');

        // Jump Logic: If we are in the user-id directory, jump straight back to the Root
        const currentParent = normalizedCurrentPath.substring(0, normalizedCurrentPath.lastIndexOf('/'));
        if (currentParent === dataDir) {
           setCurrentPath(''); 
           return;
        }

        const lastSlashIndex = normalizedCurrentPath.lastIndexOf('/');
        let parentPath = normalizedCurrentPath.substring(0, lastSlashIndex);

        if (parentPath === base || !parentPath) {
            setCurrentPath(''); 
        } else {
            setCurrentPath(parentPath);
        }
    }
  }

  const navigateToRoot = () => {
    // navigates to the initial browsable path
    setCurrentPath('')
  }

  const getBreadcrumbs = () => {
    if (!currentPath || !baseDirPathFull || !dataDirPathFull) return [] 
    
    // Normalize trailing slashes for reliable comparison
    const base = baseDirPathFull.replace(/\/$/, '');
    const dataDir = dataDirPathFull.replace(/\/$/, '');

    // Determine the path relative to the browsable root
    let relativePath = currentPath.replace(base, '');
    relativePath = relativePath.startsWith('/') ? relativePath.substring(1) : relativePath;
    relativePath = relativePath.endsWith('/') ? relativePath.slice(0, -1) : relativePath;

    const pathSegments = relativePath.split('/').filter(Boolean);
    const breadcrumbsWithPaths = [];
    let cumulativePath = base; 

    pathSegments.forEach((crumb) => {
      // Logic: A segment is the User ID if the path leading up to it is exactly the data_dir
      const isUserIdSegment = cumulativePath === dataDir;
      
      // Always update the physical path so navigation stays correct
      cumulativePath = cumulativePath + '/' + crumb;
      
      // Only push to the UI array if it's not the hidden ID segment
      if (!isUserIdSegment) {
        breadcrumbsWithPaths.push({
          name: crumb,
          path: cumulativePath,
        });
      }
    });

    return breadcrumbsWithPaths;
  }

  return (
    <div 
      className="bg-white rounded-lg shadow-md p-6 relative"
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {/* DRAG OVERLAY */}
      {isDragging && (
        <div className="absolute inset-0 z-50 bg-primary-500/10 border-4 border-dashed border-primary-500 rounded-lg flex items-center justify-center backdrop-blur-sm">
          <div className="bg-white p-6 rounded-xl shadow-xl text-center">
            <UploadIcon className="w-12 h-12 text-primary-500 mx-auto mb-2 animate-bounce" />
            <p className="text-lg font-bold text-primary-700">
              Drop to upload to: <span className="font-mono text-primary-900">
                /{getBreadcrumbs().map(b => b.name).join('/') || 'root'}
              </span>
            </p>
            <p className="text-sm text-gray-500 mt-1">
              {currentPath.includes(sharedDataDirPathFull) 
                ? 'This will be visible to everyone.' 
                : 'This will be your private data.'}
            </p>
          </div>
        </div>
      )}
      {/* HEADER */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-gray-800 flex items-center gap-2">
          <Folder className="w-5 h-5" />
          File Browser
        </h2>
        {/* ACTION BUTTONS CONTAINER */}
        <div className="flex gap-2">
          {(() => {
            const isAtRoot = currentPath === '';

            // Define base styles for disabled vs active
            const disabledClasses = "opacity-50 cursor-not-allowed pointer-events-none";

            return (
              <>
                {/* UPLOAD BUTTON */}
                <label 
                  title={isAtRoot ? "Please enter a directory to upload files" : ""}
                  className={`flex items-center gap-2 px-3 py-1.5 text-sm font-medium transition-colors shadow-sm rounded-lg ${
                    isAtRoot 
                      ? `bg-gray-100 text-gray-400 border border-gray-200 ${disabledClasses}` 
                      : "bg-primary-600 text-white hover:bg-primary-700 cursor-pointer"
                  }`}
                >
                  <UploadIcon className="w-4 h-4" />
                  Upload
                  <input
                    type="file"
                    multiple
                    className="hidden"
                    disabled={isAtRoot}
                    onChange={(e) => handleFileUpload(e.target.files, e)}
                  />
                </label>

                {/* NEW FOLDER BUTTON */}
                <button
                  disabled={isAtRoot}
                  onClick={() => setIsCreatingDirectory(true)}
                  title={isAtRoot ? "Please enter a directory to create a new folder" : ""}
                  className={`flex items-center gap-2 px-3 py-1.5 text-sm font-medium transition-colors border rounded-lg ${
                    isAtRoot
                      ? `bg-gray-50 text-gray-400 border-gray-200 ${disabledClasses}`
                      : "text-primary-600 border-primary-200 hover:bg-primary-50"
                  }`}
                >
                  <FolderPlus className="w-4 h-4" />
                  New Folder
                </button>
              </>
            );
          })()}

          {/* Delete Button */}
          {hasSelectedFiles && (
            <button
              onClick={handleDeleteSelected}
              disabled={loading}
              className="flex items-center gap-2 px-3 py-1.5 text-sm font-medium text-red-600 hover:bg-red-50 rounded-lg transition-colors border border-red-200"
            >
              <Trash2 className="w-4 h-4" />
              Delete {selectedFiles.size} Item(s)
            </button>
          )}

          {/* Select All Button (Existing) */}
          {items.length > 0 && (
            <button
            onClick={handleSelectAll}
            className="flex items-center gap-2 px-3 py-1.5 text-sm font-medium text-primary-600 hover:bg-primary-50 rounded-lg transition-colors border border-primary-200"
          >
            {(() => {
              const files = getCurrentDirectoryFiles()
              const folders = getCurrentDirectoryFolders()
              const allFilesSelected = files.length === 0 || files.every(f => isItemSelected(f))
              const allFoldersSelected = folders.length === 0 || folders.every(f => directorySelectionState.get(f.path) === true)
              return allFilesSelected && allFoldersSelected
            })() ? (
              <>
                <CheckSquare className="w-4 h-4" />
                Deselect All
              </>
            ) : (
              <>
                <Square className="w-4 h-4" />
                Select All
              </>
            )}
          </button>
          )}
        </div>
      </div>

      {/* Breadcrumb Navigation */}
      <div className="mb-4 flex items-center gap-2 text-sm">
        <button
          onClick={navigateToRoot}
          className="flex items-center gap-1 px-2 py-1 hover:bg-gray-100 rounded transition-colors"
        >
          <Home className="w-4 h-4" />
          <span>Root</span>
        </button>
        {getBreadcrumbs().map((crumb, index) => {
          return (
            <div key={index} className="flex items-center gap-2">
              <ChevronRight className="w-4 h-4 text-gray-400" />
              <button
                // Use the pre-calculated absolute path from the crumb object
                onClick={() => setCurrentPath(crumb.path)} 
                className="px-2 py-1 hover:bg-gray-100 rounded transition-colors"
              >
                {/* Use the display name property for rendering */}
                {crumb.name} 
              </button>
            </div>
          )
        })}
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 text-red-600 px-3 py-2 rounded mb-4 text-sm">
          {error}
        </div>
      )}

      {/* File List */}
      <div className="border border-gray-200 rounded-lg overflow-hidden">
        {loading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="w-8 h-8 animate-spin text-primary-500" />
          </div>
        ) : (
          <div className="divide-y divide-gray-200 max-h-96 overflow-y-auto">
            {/* Directory Creation Input */}
            {isCreatingDirectory && (
              <div className="flex items-center gap-3 px-4 py-3 bg-primary-50 transition-colors">
                <Folder className="w-5 h-5 text-primary-500 ml-7" />
                <input
                  autoFocus
                  type="text"
                  value={newDirectoryName}
                  onChange={(e) => setNewDirectoryName(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') handleCreateDirectory();
                    if (e.key === 'Escape') setIsCreatingDirectory(false);
                  }}
                  placeholder="Folder name..."
                  className="flex-1 bg-white border border-primary-300 rounded px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
                  disabled={isCreating}
                />
                <div className="flex gap-1">
                  <button 
                    onClick={handleCreateDirectory}
                    disabled={isCreating || !newDirectoryName.trim()}
                    className="p-1 hover:bg-primary-100 rounded text-green-600"
                  >
                    {isCreating ? <Loader2 className="w-4 h-4 animate-spin" /> : <Check className="w-4 h-4" />}
                  </button>
                  <button 
                    onClick={() => { setIsCreatingDirectory(false); setNewDirectoryName(''); }}
                    className="p-1 hover:bg-primary-100 rounded text-red-600"
                    disabled={isCreating}
                  >
                    <Close className="w-4 h-4" />
                  </button>
                </div>
              </div>
            )}

            {/* Back Button */}
            {currentPath && (
              <button
                onClick={navigateUp}
                className="w-full flex items-center gap-3 px-4 py-3 hover:bg-gray-50 transition-colors text-left"
              >
                <ChevronDown className="w-5 h-5 text-gray-400 transform rotate-90" />
                <span className="text-gray-600">..</span>
              </button>
            )}

            {/* Items or Empty State */}
            {items.length === 0 ? (
              <div className="text-center py-12 text-gray-500">
                <Folder className="w-12 h-12 mx-auto mb-3 text-gray-300" />
                <p>No items in this folder</p>
              </div>
            ) : (
              items.map((item, index) => (
                <div
                  key={index}
                  className="flex items-center gap-3 px-4 py-3 hover:bg-gray-50 transition-colors"
                >
                  {/* Checkbox (for both files and directories) */}
                  <input
                    type="checkbox"
                    checked={item.is_directory 
                      ? (selectedFiles.has(item.path) || directorySelectionState.get(item.path)) 
                      : isItemSelected(item)}
                    onChange={(e) => handleCheckboxChange(item, e)}
                    className="w-4 h-4 text-primary-600 rounded border-gray-300 focus:ring-primary-500"
                  />

                  {/* Icon and Name */}
                  <button
                    onClick={() => handleItemClick(item)}
                    className="flex items-center gap-2 flex-1 text-left"
                    disabled={!item.is_directory}
                    onMouseEnter={() => {
                      if (item.is_directory) {
                        checkDirectorySelection(item.path)
                      }
                    }}
                  >
                    {item.is_directory ? (
                      <>
                        <ChevronRight className="w-5 h-5 text-gray-400" />
                        <Folder className="w-5 h-5 text-primary-500" />
                      </>
                    ) : (
                      <File className="w-5 h-5 text-gray-400 ml-5" />
                    )}
                    <span className={`${item.is_directory ? 'font-medium text-gray-800' : 'text-gray-600'}`}>
                      {item.display_name}
                    </span>
                  </button>

                  {/* File Size */}
                  {!item.is_directory && item.size !== null && (
                    <span className="text-xs text-gray-400">
                      {formatFileSize(item.size)}
                    </span>
                  )}
                </div>
              ))
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default FileBrowser
