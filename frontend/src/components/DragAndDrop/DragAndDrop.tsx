import React, { useState } from 'react';
import Uppy, { UppyFile } from '@uppy/core';
import Webcam from '@uppy/webcam';
import ScreenCapture from '@uppy/screen-capture';
import ImageEditor from '@uppy/image-editor';
import { Dashboard } from '@uppy/react';

import '@uppy/core/dist/style.min.css';
import '@uppy/dashboard/dist/style.min.css';
import '@uppy/webcam/dist/style.min.css';
import '@uppy/screen-capture/dist/style.min.css';
import '@uppy/image-editor/dist/style.min.css';

interface DragAndDropProps {
  onFileChange: (newFiles: (UppyFile)[]) => void;
}

function DragAndDrop({ onFileChange }: DragAndDropProps) {
	// IMPORTANT: passing an initializer function to prevent Uppy from being reinstantiated on every render.
	const [uppy] = useState(() => {
		const uppy = new Uppy()
		.use(Webcam)
		.use(ScreenCapture)
		.use(ImageEditor);
		
		uppy.on('complete', (file) => {
		  // Trigger the callback whenever a new file is added
		  onFileChange(uppy.getFiles());
		});

		return uppy;
	  });
  
	  return <Dashboard uppy={uppy} plugins={['Webcam', 'ScreenCapture', 'ImageEditor']} />;
  }
  
  export default DragAndDrop;
