"use client"
import { SetStateAction, useEffect, useState, useRef } from 'react';
import { saveAs } from 'file-saver';
import JSZip from 'jszip';
import { UppyFile } from '@uppy/core';
import dynamic from 'next/dynamic';
const DragAndDropNoSSR = dynamic(() => import( '@/components/DragAndDrop/DragAndDrop'), {
  ssr: false,
});
const path = require('path');

function UploadPage() {
  const [inputImagesParameter, setInputImagesParameter] = useState<File[]>([]);
  const [masksParameter, setMasksParameter] = useState<(File | null)[]>([]);
  const [maskDirectories, setMaskDirectories] = useState<string[]>(['../../Mask/', '../../K/']);
  const [predictionsOutputPathParameter, setPredictionsOutputPathParameter] = useState<string>('../../Predictions/{name}{ext}');
  const [overlayedOutputPathParameter, setOverlayedOutputPathParameter] = useState<string>('../../Overlayed/{name}{ext}');
  const [areaParameter, setAreaParameter] = useState<string>('1');
  const [thresholdParameter, setThresholdParameter] = useState<string>('0.1');
  const [generateLabelledParameter, setGenerateLabelledParameter] = useState<boolean>(true);
  const [labelledOutputPathParameter, setLabelledOutputPathParameter] = useState<string>('../../Labelled/{name}{ext}');
  
  const [message, setMessage] = useState<string>('');
  const [deleteFloatingFiles, setDeleteFloatingFiles] = useState<boolean>(true);
  //const [deleteFloatingMasks, setDeleteFloatingMasks] = useState<boolean>(true);
  const [compareMasks, setCompareMasks] = useState<boolean>(true);

  const [resultData, setResultData] = useState<{filename: string, data: string}[]>([]);
  const [imagesToZip, setImagesToZip] = useState<Set<number>>(new Set());


  const [pipelines, setPipelines] = useState([]);
  const [pipelineParameter, setPipelineParameter] = useState('');
  const [login, setLogin] = useState('domicio088');
  const [password, setPassword] = useState('bA2XSoEu');

  const [metrics, setMetrics] = useState<{num_labels: number, area: number, cell_density: number, std_areas: number, mean_areas: number, coefficient_value: number, num_hexagonal: number, hexagonal_cell_ratio: number}[]>([]);
  const [customModelPipelines, setCustomModelPipelines] = useState(["Tiling",  "Resizing with padding", "Dynamic resizing with padding"]);
  const [customModelPipelineParameter, setCustomModelPipelineParameter] = useState('');
  const [targetHeightParameter, setTargetHeightParameter] = useState<string>('128');
  const [targetWidthParameter, setTargetWidthParameter] = useState<string>('128');
  const [downsamplingFactorParameter, setDownsamplingFactorParameter] = useState<string>('32');

  const [formErrors, setFormErrors] = useState<Record<string, string[]>>({});
  const [customModelParameter, setCustomModelParameter] = useState<File | null>(null);
  let [previousTaskState, setPreviousTaskState] = useState<string>('');

  const handleImageSelect = (index: number, isSelected: boolean) => {
      const newSelection = new Set(imagesToZip);
      if (isSelected) {
          newSelection.add(index);
      } else {
          newSelection.delete(index);
      }
      setImagesToZip(newSelection);
  };
  const deleteFloatingFilesRef = useRef(deleteFloatingFiles); // Because deleteFloatingFiles is a state variable, it is not updated immediately. This is a workaround to get the updated value immediately.

  useEffect(() => {
    fetchPipelines();
    deleteFloatingFilesRef.current = deleteFloatingFiles;
  }, [deleteFloatingFiles]);

  const fetchPipelines = async () => {
    try {
        const response = await fetch('http://localhost:8000/analysis/models');
        const data = await response.json();
        setPipelines(data);
    } catch (error) {
        console.error('Error fetching pipelines:', error);
    }
  };


  function getMaskPaths(originalImagePath: string) {
    const originalDir = path.dirname(originalImagePath);
    const filename = path.basename(originalImagePath);
  
    return maskDirectories.map(directory => path.join(originalDir, directory, filename));
  }
  
  const downloadSelectedImages = () => {
    const zip = new JSZip();

    imagesToZip.forEach(index => {
        const image = resultData[index];
        zip.file(image.filename, image.data, {base64: true});
    });

    let metricsString = "";
    metrics.forEach(metric => {
      metricsString += `path: ${resultData[metrics.indexOf(metric)].filename}\n`;
      metricsString += `Num Labels: ${metric.num_labels}\n`;
      metricsString += `Area: ${metric.area}\n`;
      metricsString += `Cell Density: ${metric.cell_density}\n`;
      metricsString += `Std Areas: ${metric.std_areas}\n`;
      metricsString += `Mean Areas: ${metric.mean_areas}\n`;
      metricsString += `Coefficient Value: ${metric.coefficient_value}\n`;
      metricsString += `Num Hexagonal: ${metric.num_hexagonal}\n`;
      metricsString += `Hexagonal Cell Ratio: ${metric.hexagonal_cell_ratio}\n`;
      metricsString += "\n";
    });
    zip.file("metrics.txt", metricsString);

    zip.generateAsync({type: 'blob'}).then(content => {
        saveAs(content, 'selected_images.zip');
    });
  };

    const handleInputImagesChange = async (files: (UppyFile)[]) => {
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
              break;
            }
          }
  
          if (!maskFound) {
            console.log("Mask file not found for", file.name);
            if (!deleteFloatingFilesRef.current) {
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

    setInputImagesParameter(prevFiles => [...prevFiles, ...newFiles]);
    if (compareMasks)
      setMasksParameter(prevMasks => [...prevMasks, ...newMasks]);
    else
      setMasksParameter(prevMasks => [...prevMasks, ...Array(newFiles.length).fill(null)]);
  };

  async function pollForResult(endpoint: string, interval: number) {
    let completed = false;
    while (!completed) {
      try {
        const response = await fetch(endpoint, {
          method: 'GET',
          headers: {
            'Accept': 'application/json; indent=4',
            'Authorization': 'Basic ' + btoa(login + ':' + password),
          }});

        const content = await response.json();
  
        if (response.ok) {
          setMessage('Task ' + content.data.state);
          if (content.data.state === 'success') {
            // Task is completed, handle the data
            console.log('data.metrics length:', content.data.metrics.length);
            console.log('data.results length:', content.data.results.length);
            setMetrics(content.data.metrics);
            setResultData(content.data.results); // Assuming the data contains the results
            console.log("data.metrics:", content.data.metrics);
            completed = true;  // Stop polling
            previousTaskState = '';
          } else {
            if (content.data.state === 'pending' && previousTaskState === 'started') {
                makeAnalysisRequest();
                return;
            }
            previousTaskState = content.data.state;
            // Task is not completed, wait for the suggested interval and then poll again
            await new Promise(resolve => setTimeout(resolve, interval * 1000));
          }
          
        } else {
          setMessage('Error while processing task' + content.data);
          completed = true; // Stop polling on error
        }
      } catch (error) {
        console.error('Polling failed', error);
        setMessage('Failed to get task status');
        completed = true; // Stop polling on error
      }
    }
  }

  const makeAnalysisRequest = async () => {
    if (inputImagesParameter.length === 0) {
      setMessage('Please select files to upload');
      return;
    }

    if (pipelineParameter === "upload" && !customModelParameter) {
      setMessage('Please upload a model');
      return;
    }

    const formData = new FormData();
    inputImagesParameter.forEach((file, index) => {
      formData.append('input_images', file);
      formData.append('input_paths', file.webkitRelativePath || file.name);
      // Assuming masks is an array of file or null
      const correspondingMask = masksParameter[index];

      if (correspondingMask) {
        formData.append(`masks`, correspondingMask);
      }
      else {
        formData.append('masks', "none");
      }
    });

    formData.append('generate_labelled_images', generateLabelledParameter.toString());
    if (generateLabelledParameter)
      formData.append('labelled_output_path', labelledOutputPathParameter);

    formData.append('predictions_output_path', predictionsOutputPathParameter);
    formData.append('overlayed_output_path', overlayedOutputPathParameter);
    formData.append('area_per_pixel', areaParameter);

    if (pipelineParameter === "upload") {
      formData.append('custom_model', customModelParameter || '');
      formData.append('custom_model_pipeline', customModelPipelineParameter);
      formData.append('threshold', thresholdParameter);
      formData.append('target_height', targetHeightParameter);
      formData.append('target_width', targetWidthParameter);
      formData.append('downsampling_factor', downsamplingFactorParameter);
    }
    else {
      formData.append('pipeline', pipelineParameter);
    }
    

    try {
      const response = await fetch('http://localhost:8000/analysis/', {
        headers: {
          'Accept': 'application/json; indent=4',
          'Authorization': 'Basic ' + btoa(login + ':' + password),
        },
        method: 'POST',
        body: formData,
      });

      if (response.status === 202) {
        setMessage('Task accepted');
        setFormErrors({});
        const content = await response.json();
        pollForResult(content.data.polling_endpoint, content.data.polling_interval);
        return;
      }
      if (response.status === 403) {
        setFormErrors({
          login: ['Invalid login or password'],
        });
        setMessage('Invalid login or password');
        return;
      }

      if (response.status === 400) {
        const content = await response.json();
        displayErrors(content.data);
        setFormErrors(content.data);
        return;
      }
      setMessage('Unexpected error');

    } catch (error) {
      console.error('Failed to make request:', error);
      setMessage('Failed to make request');
    }
  };

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    makeAnalysisRequest();
  };

  const displayErrors = (errors: Record<string, string[]>) => {
    for (const [field, messages] of Object.entries(errors)) {
      console.error(`Error in ${field}: ${messages.join(', ')}`);
      setMessage(`Error in ${field}: ${messages.join(', ')}`);
    }
  };


  const handleDeleteFloatingFilesChange = () => {
    setDeleteFloatingFiles(!deleteFloatingFiles);
  };

  const handleCompareMasksChange = () => {
    setCompareMasks(!compareMasks);
  };
  const handleGenerateLebelledImagesChange = () => {
    setGenerateLabelledParameter(!generateLabelledParameter);
  };

  const handleModelChange = (event: { target: { value: SetStateAction<string>; }; }) => {
    setPipelineParameter(event.target.value);
  };

  const handlePipelineChange = (event: { target: { value: SetStateAction<string>; }; }) => {
    setCustomModelPipelineParameter(event.target.value);
  };

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
    const allowedSingleFileExtensions = ['h5', 'keras', 'zip'];

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

const handleModelFilesChange = async (files: (UppyFile)[]) => {
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
    console.log("handleModelFilesChange validation failed");
    return;
  }

  if (files.length === 1) {
    const file = files[0];
    
    console.log("handleModelFilesChange: recieved a single file:", file);
    //const fileContent = await file.data.arrayBuffer();
    //console.log("handleModelFilesChange: setting type to ", files[0].type);
    setCustomModelParameter(file.data as File); // Directly handle the single file's Blob
    console.log("handleModelFilesChange: custom_model:", customModelParameter);
    return;
  }
  console.log("handleModelFilesChange recieved multiple files: ", files);

  const zip = new JSZip();
  files.forEach(file => {
    console.log("handleModelFilesChange: file relative path:", file.meta.newRelativePath);
    zip.file(file.meta.newRelativePath as string, file.data, {base64: true});
  });

  try {
    const content = await zip.generateAsync({type: 'blob'});
    
    // Convert the blob to a File
    const zipFileName = "your_zip_file_name.zip"; // Replace with your desired file name
    const zipFile = new File([content], zipFileName, {type: 'application/zip'});

    setCustomModelParameter(zipFile); // Assuming setcustom_model is a function that sets the state or otherwise stores the file
    console.log("handleModelFilesChange: customModel:", zipFile);
  } catch (error) {
    console.error("Error generating zip: ", error);
  }
}

  return (
    <div>
      <form onSubmit={handleSubmit}>
      <div>
        <label>Login:</label>
        <input 
          type="text" 
          value={login} 
          onChange={(e) => setLogin(e.target.value)} 
          style={{ color: 'black' }}
        />
        <label>Password:</label>

        <input 
          type="password" 
          value={password} 
          onChange={(e) => setPassword(e.target.value)} 
          style={{ color: 'black' }}
        />
        {formErrors.login && (
            <div className="text-red-500 text-sm">{formErrors.login.join(', ')}</div>
        )}
      </div>
        <div>
          <DragAndDropNoSSR onFileChange={handleInputImagesChange} />
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
        <div className="flex items-center space-x-2">
          <label>
              <input
                  type="checkbox"
                  checked={generateLabelledParameter}
                  onChange={handleGenerateLebelledImagesChange}
              />
              Generate labelled images
          </label>
          {formErrors.generate_labelled_images && (
            <div className="text-red-500 text-sm">{formErrors.generate_labelled_images.join(', ')}</div>
          )}
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
        <div className="flex items-center space-x-2">
          <label>Predictions relative path:</label>
          <input 
            type="text" 
            value={predictionsOutputPathParameter} 
            onChange={(e) => setPredictionsOutputPathParameter(e.target.value)} 
            style={{ color: 'black' }}
          />
          {formErrors.predictions_output_path && (
            <div className="text-red-500 text-sm">{formErrors.predictions_output_path.join(', ')}</div>
          )}
        </div>
        <div className="flex items-center space-x-2">
          <label>Overlayed images relative path:</label>
          <input 
            type="text" 
            value={overlayedOutputPathParameter} 
            onChange={(e) => setOverlayedOutputPathParameter(e.target.value)} 
            style={{ color: 'black' }}
          />
          {formErrors.overlayed_output_path && (
            <div className="text-red-500 text-sm">{formErrors.overlayed_output_path.join(', ')}</div>
          )}
        </div>
        { generateLabelledParameter && 
        <div className="flex items-center space-x-2">
          <label>Labelled image relative path</label>
          <input 
            type="text" 
            value={labelledOutputPathParameter} 
            onChange={(e) => setLabelledOutputPathParameter(e.target.value)} 
            style={{ color: 'black' }}
          />
          {formErrors.labelled_output_path && (
            <div className="text-red-500 text-sm">{formErrors.labelled_output_path.join(', ')}</div>
          )}
        </div>}
        <div className="flex items-center space-x-2">
          <label>Area in mm:</label>
          <input 
            type="text" 
            value={areaParameter}
            onChange={(e) => setAreaParameter(e.target.value)} 
            style={{ color: 'black' }}
          />
          {formErrors.area_per_pixel && (
            <div className="text-red-500 text-sm">{formErrors.area_per_pixel.join(', ')}</div>
          )}
        </div>
        <div>
          <select value={pipelineParameter} onChange={handleModelChange} style={{ color: 'black' }}>
            <option value="" style={{ color: 'black' }}>Select a Model</option>
            {pipelines.map((model, index) => (
              <option key={index} value={model} style={{ color: 'black' }}>{model}</option>
            ))}
            <option value="upload" style={{ color: 'black' }}>Upload Your Own Model</option>
          </select>

          {pipelineParameter === "upload" && (
            <div>
              <DragAndDropNoSSR onFileChange={handleModelFilesChange} />
              <p>Upload .h5, .keras files or a zipped model directory.</p>
              <select value={customModelPipelineParameter} onChange={handlePipelineChange} style={{ color: 'black' }}>
                <option value="" style={{ color: 'black' }}>Select a pipeline</option>
                {customModelPipelines.map((pipeline, index) => (
                  <option key={index} value={pipeline} style={{ color: 'black' }}>{pipeline}</option>
                ))}
              </select>
              <div className="flex items-center space-x-2">
                <label>Binarization threshold:</label>
                <input
                  type="text" 
                  value={thresholdParameter}
                  onChange={(e) => setThresholdParameter(e.target.value)} 
                  style={{ color: 'black' }}
                />
                {formErrors.threshold && (
                  <div className="text-red-500 text-sm">{formErrors.threshold.join(', ')}</div>
                )}
              </div>
              {(customModelPipelineParameter === "Tiling" || customModelPipelineParameter === "Resizing with padding") && (
                <div>
                  <div className="flex items-center space-x-2">
                    <label>Target height:</label>
                    <input
                      type="text" 
                      value={targetHeightParameter}
                      onChange={(e) => setTargetHeightParameter(e.target.value)}
                      style={{ color: 'black' }}
                    />
                    {formErrors.target_height && (
                      <div className="text-red-500 text-sm">{formErrors.target_height.join(', ')}</div>
                    )}
                  </div>
                  <div className="flex items-center space-x-2">
                    <label>Target width:</label>
                    <input
                      type="text" 
                      value={targetWidthParameter}
                      onChange={(e) => setTargetWidthParameter(e.target.value)}
                      style={{ color: 'black' }}
                    />
                    {formErrors.target_width && (
                      <div className="text-red-500 text-sm">{formErrors.target_width.join(', ')}</div>
                    )}
                  </div>
                </div>
                )}
                {customModelPipelineParameter === "Dynamic resizing with padding" && (
                <div>
                  <label htmlFor="downsamplingFactor">Downsampling factor:</label>
                  <div className="flex items-center space-x-2">
                    <input
                      id="downsamplingFactor"
                      type="text"
                      value={downsamplingFactorParameter}
                      onChange={(e) => setDownsamplingFactorParameter(e.target.value)}
                      className="text-black"
                    />
                    {formErrors.downsampling_factor && (
                      <div className="text-red-500 text-sm">{formErrors.downsampling_factor.join(', ')}</div>
                    )}
                  </div>
                </div>
                )}
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
      {resultData.map((image, index) => {
    // This calculation finds the corresponding metrics index
    // for every group of 3 images.
    const numImagesPerPrediction = (generateLabelledParameter ? 3 : 2);
    const apiDataIndex = Math.floor(index / numImagesPerPrediction);

    return (
        <div key={index}>
            {index % numImagesPerPrediction === 0 && (
                <div>
                    <p>Num Labels: {metrics[apiDataIndex]?.num_labels}</p>
                    <p>Area: {metrics[apiDataIndex]?.area}</p>
                    <p>Cell Density: {metrics[apiDataIndex]?.cell_density}</p>
                    <p>Std Areas: {metrics[apiDataIndex]?.std_areas}</p>
                    <p>Mean Areas: {metrics[apiDataIndex]?.mean_areas}</p>
                    <p>Coefficient Value: {metrics[apiDataIndex]?.coefficient_value}</p>
                    <p>Num Hexagonal: {metrics[apiDataIndex]?.num_hexagonal}</p>
                    <p>Percentage of hexagonal cells: {metrics[apiDataIndex]?.hexagonal_cell_ratio}</p>
                </div>
            )}

            {/* Displaying Image */}
            <img src={`data:image/png;base64,${image.data}`} alt={`Image ${index}`} />
            
            {/* Checkbox for selection */}
            <input
                type="checkbox"
                checked={imagesToZip.has(index)}
                onChange={(e) => handleImageSelect(index, e.target.checked)}
            />
        </div>
    );
})}
      </div>
    </div>
  );
  
}



export default UploadPage;
