"use client"
import { useState } from 'react';
const path = require('path');

function UploadPage() {
  const [files, setFiles] = useState<File[]>([]);
  const [masks, setMasks] = useState<(File | null)[]>([]);
  const [maskDirectory, setMaskDirectory] = useState<string>('../../Mask/'); // Added state for mask directory
  const [predictionsPath, setPredictionsPath] = useState<string>('../../Predictions/{name}{ext}');
  const [overlayedPath, setOverlayedPath] = useState<string>('../../Overlayed/{name}{ext}');
  const [areaParameter, setAreaParameter] = useState<string>('1');
  
  
  const [message, setMessage] = useState<string>('');
  const [deleteFloatingFiles, setDeleteFloatingFiles] = useState<boolean>(true);
  //const [deleteFloatingMasks, setDeleteFloatingMasks] = useState<boolean>(true);
  const [compareMasks, setCompareMasks] = useState<boolean>(true);

  function getMaskPath(originalImagePath: string) {
    const originalDir = path.dirname(originalImagePath);
    const filename = path.basename(originalImagePath);
    const newPath = path.join(originalDir, maskDirectory, filename); // Use maskDirectory here
    return newPath;
  }

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      const fileArray = Array.from(event.target.files);
      // copy fileArray
      const fileArrayCopy = [...fileArray];
  
      // Initialize an array to store the mask files
      const newFiles : File[] = [];
      const newMasks : (File | null)[] = [];
  
      // Process each file
      for (const file of fileArray) {
        // Check if the file is an image
        if (file.type.startsWith('image/')) {
          // Construct the path to the corresponding mask file
          const maskPath = getMaskPath(file.webkitRelativePath || file.name, '../../Mask/');
          console.log("====================================");
          console.log(file.webkitRelativePath);
          console.log(maskPath);

          
          // Check if the mask file exists in the file list
          const maskFile = fileArray.find(f => f.webkitRelativePath === maskPath);
          if (maskFile) {
            console.log("Found mask file");
            if (maskFile === file) {
              console.log("Found mask instantly");
              continue;
            }
            
            // Add the mask file to the newMasks array
            newMasks.push(maskFile);
            newFiles.push(file);
          }
          else {
            console.log("Mask file not found");
            if (!deleteFloatingFiles)
            {
              newFiles.push(file);
              newMasks.push(null);
            }
          }
          console.log("====================================");
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
          <label>Mask Directory:</label>
          <input 
            type="text" 
            value={maskDirectory} 
            onChange={(e) => setMaskDirectory(e.target.value)} 
            style={{ color: 'black' }}
          />
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
