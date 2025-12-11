/**
 * Video Preview Module
 * 
 * Handles video file selection and preview functionality.
 */

/**
 * Handle video file selection - extract first frame for preview
 * @param {Event} event - File input change event
 */
function handleVideoFileSelection(event) {
  const file = event.target.files[0];
  const btnExtractFrames = document.getElementById('btn-extract-frames');

  if (!file) {
    const previewContainer = document.getElementById('video-preview-container');
    if (previewContainer) {
      previewContainer.style.display = 'none';
    }
    if (btnExtractFrames) {
      btnExtractFrames.disabled = true;
    }
    return;
  }

  // Enable button when file is selected
  if (btnExtractFrames) {
    btnExtractFrames.disabled = false;
  }

  // Hide drag-drop-zone when video is selected
  const dragDropZone = document.getElementById('drag-drop-zone');
  if (dragDropZone) {
    dragDropZone.style.display = 'none';
  }

  // Show loading state
  const container = document.getElementById('video-preview-container');
  if (container) {
    container.style.display = 'block';
    const metadataGrid = document.getElementById('video-metadata-grid');
    if (metadataGrid) {
      metadataGrid.style.display = 'grid';
    }
    const previewFilename = document.getElementById('preview-filename');
    if (previewFilename) {
      previewFilename.textContent = file.name;
    }

    // Clear video preview
    const videoPreview = document.getElementById('video-preview');
    if (videoPreview) {
      videoPreview.src = '';
    }

    // Clear first frame preview (hidden image for hold detection)
    const firstFramePreview = document.getElementById('first-frame-preview');
    if (firstFramePreview) {
      firstFramePreview.src = '';
    }
  }

  // Create blob URL for video file and set it to video preview element
  const fileUrl = URL.createObjectURL(file);
  const videoPreview = document.getElementById('video-preview');
  if (videoPreview) {
    videoPreview.src = fileUrl;
    videoPreview.load();
  }

  // Store video URL for cleanup
  WorkflowState.setVideoPreviewUrl(fileUrl);

  try {
    // Extract first frame using video element and canvas (for hold detection)
    const video = document.createElement('video');
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    // Set crossOrigin for blob URLs
    video.crossOrigin = 'anonymous';

    let metadataLoaded = false;

    video.onloadedmetadata = () => {
      metadataLoaded = true;
      // Seek to first frame
      video.currentTime = 0;
    };

    video.onseeked = () => {
      if (!metadataLoaded) return;

      // Draw frame to canvas
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0);

      // Convert to blob and create preview URL
      canvas.toBlob(blob => {
        if (blob) {
          const previewUrl = URL.createObjectURL(blob);
          WorkflowState.setFirstFrameImageUrl(previewUrl); // Store for hold detection

          // Set to hidden image element (for hold detection UI)
          const firstFramePreview = document.getElementById('first-frame-preview');
          if (firstFramePreview) {
            firstFramePreview.src = previewUrl;
          }

          // Also display in hold detection UI
          const holdLabelingFrame = document.getElementById('hold-labeling-frame');
          if (holdLabelingFrame) {
            holdLabelingFrame.src = previewUrl;
          }

          // Show hold labeling UI
          const holdLabelingUI = document.getElementById('hold-labeling-ui');
          if (holdLabelingUI) {
            holdLabelingUI.style.display = 'block';
          }
        }
      }, 'image/jpeg', 0.8);

      video.pause();
    };

    // Load video file for first frame extraction
    video.src = fileUrl;

    // Add error handling
    video.onerror = () => {
      console.error('Failed to load video file for first frame extraction');
    };

    video.load();
  } catch (error) {
    console.error('Error processing video file:', error);
  }

  // Update metadata display
  updateVideoPreview();
}

/**
 * Update video preview metadata (hold color and route difficulty)
 */
function updateVideoPreview() {
  const holdColorSelect = document.getElementById('hold-color');
  const routeDifficultySelect = document.getElementById('route-difficulty');

  if (holdColorSelect) {
    WorkflowState.setHoldColor(holdColorSelect.value);
  }
  if (routeDifficultySelect) {
    WorkflowState.setRouteDifficulty(routeDifficultySelect.value);
  }
}

/**
 * Clear video selection and restore drag-drop zone
 */
function clearVideoSelection() {
  // Show drag-drop-zone again
  const dragDropZone = document.getElementById('drag-drop-zone');
  if (dragDropZone) {
    dragDropZone.style.display = 'block';
  }

  // Hide video preview container
  const previewContainer = document.getElementById('video-preview-container');
  if (previewContainer) {
    previewContainer.style.display = 'none';
  }

  // Hide metadata grid
  const metadataGrid = document.getElementById('video-metadata-grid');
  if (metadataGrid) {
    metadataGrid.style.display = 'none';
  }

  // Clear file input
  const videoFileInput = document.getElementById('video-file');
  if (videoFileInput) {
    videoFileInput.value = '';
  }

  // Disable extract frames button
  const btnExtractFrames = document.getElementById('btn-extract-frames');
  if (btnExtractFrames) {
    btnExtractFrames.disabled = true;
  }

  // Clear video preview
  const videoPreview = document.getElementById('video-preview');
  if (videoPreview) {
    videoPreview.src = '';
    videoPreview.load(); // Reset video element
  }

  // Clear preview image (hidden, for hold detection)
  const firstFramePreview = document.getElementById('first-frame-preview');
  if (firstFramePreview) {
    firstFramePreview.src = '';
  }

  // Clear preview filename
  const previewFilename = document.getElementById('preview-filename');
  if (previewFilename) {
    previewFilename.textContent = '';
  }

  // Clear blob URLs to prevent memory leaks
  if (WorkflowState.videoPreviewUrl) {
    URL.revokeObjectURL(WorkflowState.videoPreviewUrl);
    WorkflowState.setVideoPreviewUrl(null);
  }
  if (WorkflowState.firstFrameImageUrl) {
    URL.revokeObjectURL(WorkflowState.firstFrameImageUrl);
    WorkflowState.setFirstFrameImageUrl(null);
  }

  // Reset workflow state
  WorkflowState.setCurrentUploadId(null);
  WorkflowState.setCurrentVideoName(null);
  WorkflowState.setFirstFrameImageUrl(null);
  WorkflowState.setVideoPreviewUrl(null);
}
