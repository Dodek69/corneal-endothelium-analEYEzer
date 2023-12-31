"use client"
import { SetStateAction, useEffect, useState } from 'react';
import { saveAs } from 'file-saver';
import JSZip from 'jszip';
import { UppyFile } from '@uppy/core';
import dynamic from 'next/dynamic';
const DragAndDropNoSSR = dynamic(() => import( '@/components/DragAndDrop/DragAndDrop'), {
  ssr: false,
});
const path = require('path');

function UploadPage() {
  const [files, setFiles] = useState<File[]>([]);
  const [masks, setMasks] = useState<(File | null)[]>([]);
  const [maskDirectories, setMaskDirectories] = useState<string[]>(['../../Mask/', '../../K/']);
  const [predictionsPath, setPredictionsPath] = useState<string>('../../Predictions/{name}{ext}');
  const [overlayedPath, setOverlayedPath] = useState<string>('../../Overlayed/{name}{ext}');
  const [areaParameter, setAreaParameter] = useState<string>('1');
  const [generateLabelledImages, setGenerateLebelledImage] = useState<boolean>(true);
  const [labelledImagesPath, setLabelledImagePath] = useState<string>('../../Labelled/{name}{ext}');
  
  const [message, setMessage] = useState<string>('');
  const [deleteFloatingFiles, setDeleteFloatingFiles] = useState<boolean>(true);
  //const [deleteFloatingMasks, setDeleteFloatingMasks] = useState<boolean>(true);
  const [compareMasks, setCompareMasks] = useState<boolean>(true);


  const [imageData, setImageData] = useState<{filename: string, data: string}[]>([]);
  const [selectedImages, setSelectedImages] = useState<Set<number>>(new Set());


  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');

  const handleImageSelect = (index: number, isSelected: boolean) => {
      const newSelection = new Set(selectedImages);
      if (isSelected) {
          newSelection.add(index);
      } else {
          newSelection.delete(index);
      }
      setSelectedImages(newSelection);
  };

  useEffect(() => {
    fetchModels();
  }, []);


  const fetchModels = async () => {
    try {
        const response = await fetch('http://localhost:8000/analysis/');
        const data = await response.json();
        setModels(data);
    } catch (error) {
        console.error('Error fetching parameters:', error);
    }
};

  function getMaskPaths(originalImagePath: string) {
    const originalDir = path.dirname(originalImagePath);
    const filename = path.basename(originalImagePath);
  
    return maskDirectories.map(directory => path.join(originalDir, directory, filename));
  }
  
  const downloadSelectedImages = () => {
    const zip = new JSZip();

    selectedImages.forEach(index => {
        const image = imageData[index];
        zip.file(image.filename, image.data, {base64: true});
    });

    zip.generateAsync({type: 'blob'}).then(content => {
        saveAs(content, 'selected_images.zip');
    });
  };

    const handleFileChange = async (files: (UppyFile)[]) => {
      if (files.length === 0) {
        setMessage('Please select a file or folder to upload');
        return;
      }
  
      // Initialize arrays to store the files and masks
      const newFiles: File[] = [];
      const newMasks: (File | null)[] = [];
  
      for (const file of files) {
        if (file.data.type.startsWith('image/')) {
          // Get an array of possible mask paths
          const maskPaths = getMaskPaths(file.meta.relativePath as string || file.name);
          console.log("maskPaths:", maskPaths);
          let maskFound = false;
          for (const maskPath of maskPaths) {
            const maskFile = files.find(f => f.meta.relativePath === maskPath);
            if (maskFile) {
              console.log("Found mask file at:", maskPath);
              newMasks.push(maskFile.data as File);
              newFiles.push(file.data as File);
              maskFound = true;
              break; // Stop searching once a mask is found
            }
          }
  
          if (!maskFound) {
            console.log("Mask file not found for", file.name);
            if (!deleteFloatingFiles) {
              newFiles.push(file.data as File);
              newMasks.push(null);
            }
          }
        }
      }

    // console log length of fileArray and newMasks
    console.log("====================================");
    console.log(newFiles.length);
    console.log(newMasks.length);
    console.log("====================================");

    setFiles(prevFiles => [...prevFiles, ...newFiles]);
    if (compareMasks)
      setMasks(prevMasks => [...prevMasks, ...newMasks]);
    else
      setMasks(prevMasks => [...prevMasks, ...Array(newFiles.length).fill(null)]);
  };

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (files.length === 0) {
      setMessage('Please select a file or folder to upload');
      return;
    }

    if (!uploadedModel && !selectedModel) {
      setMessage('No model');
      return;
    }
    const formData = new FormData();
    files.forEach((file, index) => {
      formData.append('files', file);
      formData.append('paths', file.webkitRelativePath || file.name);
      // Assuming masks is an array of file or null
      const correspondingMask = masks[index];

      if (correspondingMask) {
        formData.append(`masks`, correspondingMask);
      }
      else {
        formData.append(`masks`, new File([], ''));
      }
    });

    formData.append('model', selectedModel);
    // uploadedModel or None
    console.log("sending uploadedModel:", uploadedModel);
    formData.append('uploadedModel', uploadedModel || '');
    formData.append('predictionsPath', predictionsPath);
    formData.append('overlayedPath', overlayedPath);
    formData.append('areaParameter', areaParameter);
    formData.append('generateLabelledImages', generateLabelledImages.toString());
    formData.append('labelledImagesPath', labelledImagesPath);
    try {
      const response = await fetch('http://localhost:8000/analysis/', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        setImageData(data);
      } else {
        setMessage('Failed to upload file(s)');
      }
    } catch (error) {
      console.error('Upload failed', error);
      setMessage('Failed to upload file(s)');
    }
  };
  const handleDeleteFloatingFilesChange = () => {
    setDeleteFloatingFiles(!deleteFloatingFiles);
  };

  const handleCompareMasksChange = () => {
    setCompareMasks(!compareMasks);
  };
  const handleGenerateLebelledImageChange = () => {
    setGenerateLebelledImage(!generateLabelledImages);
  };

  const [uploadedModel, setUploadedModel] = useState<File | null>(null);

  const handleModelChange = (event: { target: { value: SetStateAction<string>; }; }) => {
    setSelectedModel(event.target.value);
  };

  // create a function that validates the model files

  function validateSavedModelFormatFiles(files: UppyFile[]): boolean {
    let hasSavedModelPb = false;
    let hasVariablesIndex = false;
    let hasVariablesData = false;

    files.forEach(file => {
      const filename = file.meta.newRelativePath as string;
      console.log("validateSavedModelFormatFiles: file relative path:", filename);
        if (filename === 'saved_model.pb') {
            hasSavedModelPb = true;
        } else if (filename.startsWith('variables/')) {
          if (filename.endsWith('.index'))
          {
            hasVariablesIndex = true;
          }
          else if (filename.match(/variables\.data-\d+?-of-\d+?$/)) {
            hasVariablesData = true;
          }
        }
    });
    if (!hasSavedModelPb) {
      console.log('validateSavedModelFormatFiles: No saved_model.pb');
      return false;
    }
    if (!hasVariablesIndex) {
      console.log('validateSavedModelFormatFiles: No variables index');
      return false;
    }
    if (!hasVariablesData) {
      console.log('validateSavedModelFormatFiles: No variables data');
      return false;
    }
    return true;
}

const validateModelFiles = (files: UppyFile[]): boolean => {
  if (files.length === 0) {
    console.log('validateModelFiles: No files');
    return false;
  }
  if (files.length === 1) {
    const allowedSingleFileExtensions = ['h5', 'keras'];

    const file = files[0];
    const extension = file.extension.toLowerCase();
    console.log("validateModelFiles: analyzing file: ", file.meta.newRelativePath);
    console.log("validateModelFiles: file extension: ", extension);

    if (!allowedSingleFileExtensions.includes(extension)) {
      console.log(`validateModelFiles: Invalid file extension: ${extension}`);
      return false;
    }
    return true;
  }
  else {
    return validateSavedModelFormatFiles(files);
  }
};

const handleModelFiles = async (files: (UppyFile)[]) => {
    
    if (files.length > 1) {
      files.forEach(file => {
        const relativePath = file.meta.relativePath as string;
        console.log("handleModelFiles: file relative path:", relativePath);
        let parts = relativePath.split("/");  // Split the path by '/'
        parts.shift();  // Remove the first element (the folder name)
        let newPath = parts.join("/");  // Join the remaining parts back together
      
        console.log("handleModelFiles: file new path:", newPath);  // This will output 'fingerprint.pb'
        file.meta.newRelativePath = newPath;
      });
      console.log("handleModelFiles: processed relative path files:", files);
    }

    if (!validateModelFiles(files)) {
      console.log("handleModelFiles validation failed");
      return;
    }

    if (files.length === 1) {
      const file = files[0];
      
      console.log("handleModelFiles: recieved a single file:", file);
      //const fileContent = await file.data.arrayBuffer();
      //console.log("handleModelFiles: setting type to ", files[0].type);
      setUploadedModel(file.data as File); // Directly handle the single file's Blob
      console.log("handleModelFiles: uploadedModel:", uploadedModel);
      return;
    }
    console.log("handleModelFiles recieved multiple files: ", files);

    const zip = new JSZip();
    files.forEach(file => {
      console.log("handleModelFiles: file relative path:", file.meta.newRelativePath);
      zip.file(file.meta.newRelativePath as string, file.data, {base64: true});
    });

    try {
      const content = await zip.generateAsync({type: 'blob'});
      
      // Convert the blob to a File
      const zipFileName = "your_zip_file_name.zip"; // Replace with your desired file name
      const zipFile = new File([content], zipFileName, {type: 'application/zip'});

      setUploadedModel(zipFile); // Assuming setUploadedModel is a function that sets the state or otherwise stores the file
      console.log("handleModelFiles: uploadedModel:", zipFile);
    } catch (error) {
      console.error("Error generating zip: ", error);
    }
  }

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <div>
          <DragAndDropNoSSR onFileChange={handleFileChange} />
        </div>
        <div>
            <label>
                <input
                    type="checkbox"
                    checked={deleteFloatingFiles}
                    onChange={handleDeleteFloatingFilesChange}
                />
                Delete files without masks
            </label>
        </div>
        <div>
            <label>
                <input
                    type="checkbox"
                    checked={compareMasks}
                    onChange={handleCompareMasksChange}
                />
                Compare with masks
            </label>
        </div>
        <div>
            <label>
                <input
                    type="checkbox"
                    checked={generateLabelledImages}
                    onChange={handleGenerateLebelledImageChange}
                />
                Generate labelled images
            </label>
        </div>
        <div>
          <label>Mask Directories:</label>
          {maskDirectories.map((directory, index) => (
            <input 
              key={index}
              type="text" 
              value={directory} 
              onChange={(e) => {
                const newDirectories = [...maskDirectories];
                newDirectories[index] = e.target.value;
                setMaskDirectories(newDirectories);
              }} 
              style={{ color: 'black' }}
            />
          ))}
          <button onClick={() => setMaskDirectories([...maskDirectories, ''])}>
            Add More
          </button>
        </div>
        <div>
          <label>Predictions relative path:</label>
          <input 
            type="text" 
            value={predictionsPath} 
            onChange={(e) => setPredictionsPath(e.target.value)} 
            style={{ color: 'black' }}
          />
        </div>
        <div>
          <label>Overlayed images relative path:</label>
          <input 
            type="text" 
            value={overlayedPath} 
            onChange={(e) => setOverlayedPath(e.target.value)} 
            style={{ color: 'black' }}
          />
        </div>
        { generateLabelledImages && 
        <div>
          <label>Labelled image relative path</label>
          <input 
            type="text" 
            value={labelledImagesPath} 
            onChange={(e) => setLabelledImagePath(e.target.value)} 
            style={{ color: 'black' }}
          />
        </div>}
        <div>
          <label>Area in mm:</label>
          <input 
            type="text" 
            value={areaParameter}
            onChange={(e) => setAreaParameter(e.target.value)} 
            style={{ color: 'black' }}
          />
        </div>
        <div>
          <select value={selectedModel} onChange={handleModelChange} style={{ color: 'black' }}>
            <option value="" style={{ color: 'black' }}>Select a Model</option>
            {models.map((model, index) => (
              <option key={index} value={model} style={{ color: 'black' }}>{model}</option>
            ))}
            <option value="upload" style={{ color: 'black' }}>Upload Your Own Model</option>
          </select>

          {selectedModel === "upload" && (
            <div>
              <DragAndDropNoSSR onFileChange={handleModelFiles} />
              <p>Upload .h5, .keras files or a zipped model directory.</p>
          </div>
          )}
        </div>
        <button type="submit">Upload</button>
      </form>
      {message && <p>{message}</p>}
      <div>
        {/* Image display code */}
        <button onClick={downloadSelectedImages}>Download Selected Images</button>
      </div>
      <div>
        {imageData.map((image, index) => (
            <div key={index}>
                <img src={`data:image/png;base64,${image.data}`} alt={image.filename} />
                <input
                    type="checkbox"
                    checked={selectedImages.has(index)}
                    onChange={(e) => handleImageSelect(index, e.target.checked)}
                />
            </div>
        ))}
      </div>
    </div>
  );
  
}



export default UploadPage;
