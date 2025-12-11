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

  // Show loading state
  const container = document.getElementById('video-preview-container');
  if (container) {
    container.style.display = 'block';
    const previewFilename = document.getElementById('preview-filename');
    if (previewFilename) {
      previewFilename.textContent = file.name;
    }
    const firstFramePreview = document.getElementById('first-frame-preview');
    if (firstFramePreview) {
      firstFramePreview.src = '';
    }
  }

  try {
    // Extract first frame using video element and canvas
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

          const previewImg = document.getElementById('first-frame-preview');
          if (previewImg) {
            previewImg.src = previewUrl;
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

    // Load video file
    const fileUrl = URL.createObjectURL(file);
    video.src = fileUrl;

    // Add error handling
    video.onerror = () => {
      console.error('Failed to load video file');
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
