/**
 * Workflow Steps Module
 * 
 * Handles each step of the workflow process.
 */

/**
 * Step 1: Start analyzing - extract frames and begin hold detection workflow
 */
async function startAnalyzing() {
  const videoFile = document.getElementById('video-file').files[0];
  if (!videoFile) {
    showStatus('step-1', 'Please select a video file', 'error');
    return;
  }

  showStatus('step-1', 'Uploading video and extracting frames...', 'info');
  setStepActive('step-1');

  try {
    const formData = new FormData();
    formData.append('video', videoFile);

    const response = await fetch('/api/workflow/extract-frames', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Failed to extract frames: ${response.statusText}`);
    }

    const data = await response.json();
    WorkflowState.setCurrentFrameDir(data.frame_directory);

    // Use upload_id and video_name from response (or extract from path)
    const uploadId = data.upload_id || (() => {
      const pathParts = data.frame_directory.split('/');
      const dataIdx = pathParts.indexOf('data');
      // After 'data', the next part should be the video directory name
      return dataIdx >= 0 && dataIdx < pathParts.length - 1 ? pathParts[dataIdx + 1] : null;
    })();
    const videoName = data.video_name || (() => {
      // Extract video name from upload_id (format: IMG_3708_251209_125416PM)
      if (uploadId) {
        const parts = uploadId.split('_');
        if (parts.length >= 2) {
          return parts[0] + '_' + parts[1];  // e.g., "IMG_3708"
        }
        return uploadId.split('_')[0];
      }
      return null;
    })();

    // Store for later use
    WorkflowState.setCurrentUploadId(uploadId);
    WorkflowState.setCurrentVideoName(videoName);
    WorkflowState.setCurrentFrameDir(data.frame_directory);

    showStatus('step-1', `Extracted ${data.frame_count} frames`, 'success');
    setStepCompleted('step-1');
    updateDashboardStatus();

    // Automatically navigate to Step 2
    navigateToStep('step-2');

    // Show hold labeling UI and automatically start SAM segmentation
    if (typeof showHoldLabelingUI === 'function') {
      showHoldLabelingUI(uploadId, videoName, data.frame_count);
    }

  } catch (error) {
    console.error('Frame extraction failed:', error);
    showStatus('step-1', `Error: ${error.message}`, 'error');
  }
}

/**
 * Step 2: Create labeling session (currently not used in main workflow)
 */
async function createSession() {
  const sessionNameInput = document.getElementById('session-name');
  if (!sessionNameInput) {
    showStatus('step-2', 'Session name input not found', 'error');
    return;
  }
  
  const sessionName = sessionNameInput.value.trim();
  if (!sessionName) {
    showStatus('step-2', 'Please enter a session name', 'error');
    return;
  }

  if (!WorkflowState.getCurrentFrameDir()) {
    showStatus('step-2', 'No frame directory available', 'error');
    return;
  }

  showStatus('step-2', 'Creating labeling session...', 'info');

  try {
    const response = await fetch('/api/labeling/sessions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: sessionName,
        frame_dir: WorkflowState.getCurrentFrameDir(),
        use_sam: true,  // Enable SAM segmentation
        sam_checkpoint: null,  // Will use default checkpoint
      }),
    });

    if (!response.ok) {
      throw new Error(`Failed to create session: ${response.statusText}`);
    }

    const data = await response.json();
    WorkflowState.setCurrentSessionId(data.session_id);

    showStatus('step-2', 'Session created! Opening labeling UI...', 'success');

    // Open labeling UI in new tab
    setTimeout(() => {
      window.open(`/labeling?session_id=${data.session_id}`, '_blank');
    }, 1000);

    setStepCompleted('step-2');

  } catch (error) {
    console.error('Session creation failed:', error);
    showStatus('step-2', `Error: ${error.message}`, 'error');
  }
}

/**
 * Step 3: Start YOLO training
 */
async function startTraining() {
  if (!WorkflowState.getCurrentSessionId()) {
    showStatus('step-3', 'No labeling session selected', 'error');
    return;
  }

  const epochsInput = document.getElementById('train-epochs');
  const batchInput = document.getElementById('train-batch');
  
  if (!epochsInput || !batchInput) {
    showStatus('step-3', 'Training parameters not found', 'error');
    return;
  }

  const epochs = parseInt(epochsInput.value);
  const batch = parseInt(batchInput.value);

  showStatus('step-3', 'Starting YOLO training...', 'info');

  try {
    // First, export the session to YOLO format
    const exportResponse = await fetch(`/api/labeling/sessions/${WorkflowState.getCurrentSessionId()}/export`, {
      method: 'POST',
    });

    if (!exportResponse.ok) {
      throw new Error(`Failed to export dataset: ${exportResponse.statusText}`);
    }

    const exportData = await exportResponse.json();
    showStatus('step-3', `Exported ${exportData.exported_count} images. Starting training...`, 'info');

    // Start training
    const trainResponse = await fetch('/api/yolo/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dataset_yaml: 'data/holds_training/dataset.yaml',
        model: 'yolov8n.pt',
        epochs: epochs,
        batch: batch,
        imgsz: 640,
        upload_to_gcs: false, // Upload manually in Step 4
      }),
    });

    if (!trainResponse.ok) {
      throw new Error(`Failed to start training: ${trainResponse.statusText}`);
    }

    const trainData = await trainResponse.json();
    WorkflowState.setCurrentTrainingJobId(trainData.job_id);

    showStatus('step-3', `Training started (Job ID: ${trainData.job_id})`, 'success');
    setStepCompleted('step-3');
    setStepActive('step-4');

    const btnUploadGCS = document.getElementById('btn-upload-gcs');
    if (btnUploadGCS) {
      btnUploadGCS.disabled = true; // Enable after training completes
    }
    
    if (typeof loadTrainingJobs === 'function') {
      loadTrainingJobs();
    }

  } catch (error) {
    console.error('Training failed:', error);
    showStatus('step-3', `Error: ${error.message}`, 'error');
  }
}

/**
 * Step 4: Upload model to GCS
 */
async function uploadToGCS() {
  if (!WorkflowState.getCurrentTrainingJobId()) {
    showStatus('step-4', 'No training job selected', 'error');
    return;
  }

  showStatus('step-4', 'Uploading model to GCS...', 'info');

  try {
    const response = await fetch(`/api/yolo/train/jobs/${WorkflowState.getCurrentTrainingJobId()}/upload`, {
      method: 'POST',
    });

    if (!response.ok) {
      throw new Error(`Failed to upload: ${response.statusText}`);
    }

    const data = await response.json();

    if (data.status === 'already_uploaded') {
      showStatus('step-4', `Already uploaded: ${data.gcs_uri}`, 'info');
    } else {
      showStatus('step-4', `Uploaded to: ${data.gcs_uri}`, 'success');
    }

    setStepCompleted('step-4');

  } catch (error) {
    console.error('Upload failed:', error);
    showStatus('step-4', `Error: ${error.message}`, 'error');
  }
}
