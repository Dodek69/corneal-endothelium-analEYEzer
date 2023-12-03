"use client"
import { useState } from 'react';
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

  function getMaskPaths(originalImagePath: string) {
    const originalDir = path.dirname(originalImagePath);
    const filename = path.basename(originalImagePath);
  
    return maskDirectories.map(directory => path.join(originalDir, directory, filename));
  }
  

    const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
      if (event.target.files) {
        const fileArray = Array.from(event.target.files);
    
        // Initialize arrays to store the files and masks
        const newFiles: File[] = [];
        const newMasks: (File | null)[] = [];
    
        for (const file of fileArray) {
          if (file.type.startsWith('image/')) {
            // Get an array of possible mask paths
            const maskPaths = getMaskPaths(file.webkitRelativePath || file.name);
            console.log("maskPaths:", maskPaths);
            let maskFound = false;
            for (const maskPath of maskPaths) {
              const maskFile = fileArray.find(f => f.webkitRelativePath === maskPath);
              if (maskFile) {
                console.log("Found mask file at:", maskPath);
                newMasks.push(maskFile);
                newFiles.push(file);
                maskFound = true;
                break; // Stop searching once a mask is found
              }
            }
    
            if (!maskFound) {
              console.log("Mask file not found for", file.name);
              if (!deleteFloatingFiles) {
                newFiles.push(file);
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
    }
  };
  

  const downloadFile = (data, filename) => {
    const url = window.URL.createObjectURL(new Blob([data]));
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', filename); // Set the file name for download
    document.body.appendChild(link);
    link.click();
    link.parentNode.removeChild(link);
  };

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (files.length === 0) {
      setMessage('Please select a file or folder to upload');
      return;
    }

    const formData = new FormData();
    files.forEach((file, index) => {
      formData.append('files', file);
      formData.append('paths', file.webkitRelativePath || file.name);
      formData.append('predictionsPath', predictionsPath);
      formData.append('overlayedPath', overlayedPath);
      formData.append('areaParameter', areaParameter);
      formData.append('generateLabelledImages', generateLabelledImages.toString());
      formData.append('labelledImagesPath', labelledImagesPath);
      
      // Assuming masks is an array of file or null
      const correspondingMask = masks[index];

      if (correspondingMask) {
        formData.append(`masks`, correspondingMask);
      }
      else {
        formData.append(`masks`, new File([], ''));
      }
    });

    

    try {
      const response = await fetch('http://localhost:8000/analysis/', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const result = await response.blob(); // Get the response as a Blob
        downloadFile(result, 'results.zip'); // Trigger file download
        setMessage('File(s) uploaded and results downloaded successfully');
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

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <div>
          <label>Select files: </label>
          <input type="file" multiple onChange={handleFileChange} />
        </div>
        <div>
          <label>Select folder: </label>
          <input type="file" webkitdirectory="true" onChange={handleFileChange} />
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
        <button type="submit">Upload</button>
      </form>
      {message && <p>{message}</p>}
    </div>
  );
}

export default UploadPage;
