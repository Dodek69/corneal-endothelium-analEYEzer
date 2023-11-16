"use client"
import { useState } from 'react';

function UploadPage() {
  const [files, setFiles] = useState<File[]>([]);
  const [message, setMessage] = useState<string>('');

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      const fileArray = Array.from(event.target.files);
      setFiles(prevFiles => [...prevFiles, ...fileArray]);
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
    files.forEach((file) => {
      formData.append('files', file);
      formData.append('paths', file.webkitRelativePath || file.name);
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
        <button type="submit">Upload</button>
      </form>
      {message && <p>{message}</p>}
    </div>
  );
}

export default UploadPage;
