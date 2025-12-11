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

    // Setup trim controls when metadata is loaded
    videoPreview.onloadedmetadata = () => {
      setupVideoTrimControls(videoPreview);
    };
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

          // Store as both original and trimmed first frame (initially same)
          WorkflowState.setOriginalFirstFrameUrl(previewUrl);
          WorkflowState.setTrimmedFirstFrameUrl(previewUrl);

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

          // Don't auto-show hold labeling UI - wait for Step 2
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

  // Hide and reset trim controls
  const trimControls = document.getElementById('video-trim-controls');
  if (trimControls) {
    trimControls.style.display = 'none';
  }

  // Hide custom video controls
  const customControls = document.getElementById('custom-video-controls');
  if (customControls) {
    customControls.style.display = 'none';
  }

  WorkflowState.setVideoTrimStart(null);
  WorkflowState.setVideoTrimEnd(null);
  WorkflowState.setVideoDuration(null);
  WorkflowState.setOriginalFirstFrameUrl(null);
  WorkflowState.setTrimmedFirstFrameUrl(null);
}

/**
 * Setup video trim controls with dual range sliders
 * @param {HTMLVideoElement} videoElement - The video element
 */
function setupVideoTrimControls(videoElement) {
  const trimControls = document.getElementById('video-trim-controls');
  const startSlider = document.getElementById('trim-start-slider');
  const endSlider = document.getElementById('trim-end-slider');
  const startTimeDisplay = document.getElementById('trim-start-time');
  const endTimeDisplay = document.getElementById('trim-end-time');
  const durationDisplay = document.getElementById('trim-duration');
  const rangeSelected = document.getElementById('trim-range-selected');
  const resetBtn = document.getElementById('btn-reset-trim');

  if (!trimControls || !startSlider || !endSlider) {
    return;
  }

  const duration = videoElement.duration;
  if (!duration || !isFinite(duration)) {
    return;
  }

  // Show trim controls
  trimControls.style.display = 'block';

  // Initialize sliders
  startSlider.max = duration;
  endSlider.max = duration;
  startSlider.value = 0;
  endSlider.value = duration;

  // Store duration in state
  WorkflowState.setVideoDuration(duration);
  WorkflowState.setVideoTrimStart(0);
  WorkflowState.setVideoTrimEnd(duration);

  // Setup custom video controls
  setupCustomVideoControls(videoElement);

  // Format time helper (mm:ss)
  function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  }

  // Update visual range indicator
  function updateRangeIndicator() {
    const startPercent = (parseFloat(startSlider.value) / duration) * 100;
    const endPercent = (parseFloat(endSlider.value) / duration) * 100;
    if (rangeSelected) {
      rangeSelected.style.left = `${startPercent}%`;
      rangeSelected.style.right = `${100 - endPercent}%`;
    }
  }

  // Update displays
  function updateTrimDisplays() {
    const startTime = parseFloat(startSlider.value);
    const endTime = parseFloat(endSlider.value);

    if (startTimeDisplay) startTimeDisplay.textContent = formatTime(startTime);
    if (endTimeDisplay) endTimeDisplay.textContent = formatTime(endTime);
    if (durationDisplay) durationDisplay.textContent = formatTime(Math.max(0, endTime - startTime));

    updateRangeIndicator();

    // Store in state
    WorkflowState.setVideoTrimStart(startTime);
    WorkflowState.setVideoTrimEnd(endTime);
  }

  // Event listeners for sliders
  startSlider.style.pointerEvents = 'auto';
  endSlider.style.pointerEvents = 'auto';

  // Smart z-index management based on click position
  // Helper to determine which slider to activate based on click position
  function getActiveSlider(event, sliderElement) {
    const rect = sliderElement.getBoundingClientRect();
    const clickX = event.clientX - rect.left;
    const sliderWidth = rect.width;

    const startVal = parseFloat(startSlider.value);
    const endVal = parseFloat(endSlider.value);

    // Calculate click position as percentage
    const clickPercent = clickX / sliderWidth;
    const clickValue = clickPercent * duration;

    // Simple logic: if click is before middle point, it's start slider
    // if click is after middle point, it's end slider
    const middlePoint = (startVal + endVal) / 2;

    return clickValue < middlePoint ? 'start' : 'end';
  }

  // Add mousedown handler to the container to detect clicks
  const sliderContainer = startSlider.parentElement;
  if (sliderContainer) {
    sliderContainer.addEventListener('mousedown', (e) => {
      // Only handle if clicking on the slider area (not other elements)
      if (e.target === startSlider || e.target === endSlider || e.target === rangeSelected) {
        return; // Let the slider's own handler deal with it
      }

      // Determine which slider to activate based on click position
      const selected = getActiveSlider(e, startSlider);

      if (selected === 'start') {
        startSlider.style.zIndex = '5';
        endSlider.style.zIndex = '3';
        // Programmatically focus and trigger the slider
        startSlider.focus();
      } else {
        endSlider.style.zIndex = '5';
        startSlider.style.zIndex = '3';
        // Programmatically focus and trigger the slider
        endSlider.focus();
      }
    }, true); // Use capture phase
  }

  // Handle start slider interactions
  let startSliderActive = false;
  let endSliderActive = false;

  startSlider.addEventListener('mousedown', () => {
    startSliderActive = true;
    startSlider.style.zIndex = '5';
    endSlider.style.zIndex = '3';
  });

  endSlider.addEventListener('mousedown', () => {
    endSliderActive = true;
    endSlider.style.zIndex = '5';
    startSlider.style.zIndex = '3';
  });

  document.addEventListener('mouseup', () => {
    if (startSliderActive || endSliderActive) {
      startSliderActive = false;
      endSliderActive = false;
    }
  });

  startSlider.addEventListener('input', () => {
    const startVal = parseFloat(startSlider.value);
    const endVal = parseFloat(endSlider.value);

    // Ensure start doesn't exceed end - 1 second
    if (startVal >= endVal - 1) {
      startSlider.value = Math.max(0, endVal - 1);
    }

    updateTrimDisplays();

    // Pause video when adjusting trim
    if (videoElement && !videoElement.paused) {
      videoElement.pause();
    }

    // Seek video to start time for preview
    if (videoElement) {
      videoElement.currentTime = parseFloat(startSlider.value);
    }

    // Update custom video controls (seek bar range changed)
    if (window._updateVideoSeekBar) {
      window._updateVideoSeekBar();
    }

    // Update trimmed first frame preview
    extractTrimmedFirstFrame(videoElement, parseFloat(startSlider.value));
  });

  endSlider.addEventListener('input', () => {
    const startVal = parseFloat(startSlider.value);
    const endVal = parseFloat(endSlider.value);

    // Ensure end doesn't go below start + 1 second
    if (endVal <= startVal + 1) {
      endSlider.value = Math.min(duration, startVal + 1);
    }

    updateTrimDisplays();

    // Pause video when adjusting trim
    if (videoElement && !videoElement.paused) {
      videoElement.pause();
    }

    // Seek video to end time for preview
    if (videoElement) {
      videoElement.currentTime = parseFloat(endSlider.value);
    }

    // Update custom video controls (seek bar range changed)
    if (window._updateVideoSeekBar) {
      window._updateVideoSeekBar();
    }
  });

  // Reset button
  if (resetBtn) {
    resetBtn.addEventListener('click', () => {
      startSlider.value = 0;
      endSlider.value = duration;
      updateTrimDisplays();
      if (videoElement) {
        videoElement.pause();
        videoElement.currentTime = 0;
      }
      // Reset trimmed frame to original
      const originalUrl = WorkflowState.getOriginalFirstFrameUrl();
      if (originalUrl) {
        WorkflowState.setTrimmedFirstFrameUrl(originalUrl);
      }
      // Update custom video controls
      if (window._updateVideoSeekBar) {
        window._updateVideoSeekBar();
      }
    });
  }

  // Initial update
  updateTrimDisplays();

  // Initialize z-index (both same, will change on interaction)
  startSlider.style.zIndex = '3';
  endSlider.style.zIndex = '3';
}

/**
 * Extract a frame at the specified time from video
 * @param {HTMLVideoElement} videoElement - Video element (can be blob URL video)
 * @param {number} timeSeconds - Time in seconds to extract frame
 */
function extractTrimmedFirstFrame(videoElement, timeSeconds) {
  if (!videoElement || !videoElement.src) return;

  const video = document.createElement('video');
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');

  video.crossOrigin = 'anonymous';

  let metadataLoaded = false;

  video.onloadedmetadata = () => {
    metadataLoaded = true;
    video.currentTime = timeSeconds;
  };

  video.onseeked = () => {
    if (!metadataLoaded) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    canvas.toBlob(blob => {
      if (blob) {
        // Revoke old trimmed frame URL
        const oldUrl = WorkflowState.getTrimmedFirstFrameUrl();
        if (oldUrl && oldUrl !== WorkflowState.getOriginalFirstFrameUrl()) {
          URL.revokeObjectURL(oldUrl);
        }

        const frameUrl = URL.createObjectURL(blob);
        WorkflowState.setTrimmedFirstFrameUrl(frameUrl);
      }
    }, 'image/jpeg', 0.9);

    video.pause();
  };

  video.src = videoElement.src;
  video.load();
}

/**
 * Setup custom video controls (play/pause, seek bar)
 * @param {HTMLVideoElement} videoElement - The video element
 */
function setupCustomVideoControls(videoElement) {
  const customControls = document.getElementById('custom-video-controls');
  const btnPlayPause = document.getElementById('btn-play-pause');
  const playIcon = document.getElementById('play-icon');
  const pauseIcon = document.getElementById('pause-icon');
  const currentTimeDisplay = document.getElementById('video-current-time');
  const totalTimeDisplay = document.getElementById('video-total-time');
  const seekBar = document.getElementById('video-seek-bar');
  const seekProgress = document.getElementById('video-seek-progress');

  if (!customControls || !videoElement) {
    return;
  }

  const duration = videoElement.duration;
  if (!duration || !isFinite(duration)) {
    return;
  }

  // Show custom controls
  customControls.style.display = 'flex';

  // Format time helper (mm:ss)
  function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  }

  // Get trim boundaries (or use full video if not set)
  function getTrimStart() {
    return WorkflowState.getVideoTrimStart() || 0;
  }

  function getTrimEnd() {
    return WorkflowState.getVideoTrimEnd() || duration;
  }

  function getTrimDuration() {
    return getTrimEnd() - getTrimStart();
  }

  // Update play/pause button icon
  function updatePlayPauseButton() {
    if (videoElement.paused) {
      playIcon.style.display = 'inline';
      pauseIcon.style.display = 'none';
    } else {
      playIcon.style.display = 'none';
      pauseIcon.style.display = 'inline';
    }
  }

  // Update seek bar and time display
  function updateSeekBar() {
    const trimStart = getTrimStart();
    const trimEnd = getTrimEnd();
    const trimDuration = getTrimDuration();

    // Calculate relative position within trim range
    const relativeTime = Math.max(0, videoElement.currentTime - trimStart);
    const progress = trimDuration > 0 ? (relativeTime / trimDuration) * 100 : 0;

    seekBar.value = progress;
    if (seekProgress) {
      seekProgress.style.width = `${progress}%`;
    }

    // Update time display (relative to trim start)
    if (currentTimeDisplay) {
      currentTimeDisplay.textContent = formatTime(relativeTime);
    }
    if (totalTimeDisplay) {
      totalTimeDisplay.textContent = formatTime(trimDuration);
    }
  }

  // Handle play/pause button click
  function handlePlayPause() {
    const trimStart = getTrimStart();
    const trimEnd = getTrimEnd();

    if (videoElement.paused) {
      // If current time is outside trim range, start from trim start
      if (videoElement.currentTime < trimStart || videoElement.currentTime >= trimEnd) {
        videoElement.currentTime = trimStart;
      }
      videoElement.play();
    } else {
      videoElement.pause();
    }
  }

  // Handle seek bar input
  function handleSeekBarInput() {
    const trimStart = getTrimStart();
    const trimDuration = getTrimDuration();

    // Convert percentage to actual time
    const relativeTime = (parseFloat(seekBar.value) / 100) * trimDuration;
    videoElement.currentTime = trimStart + relativeTime;

    // Update progress display
    if (seekProgress) {
      seekProgress.style.width = `${seekBar.value}%`;
    }
  }

  // Handle video time update (enforce trim boundaries)
  function handleTimeUpdate() {
    const trimEnd = getTrimEnd();

    // Stop at trim end
    if (videoElement.currentTime >= trimEnd) {
      videoElement.pause();
      videoElement.currentTime = trimEnd - 0.01; // Slightly before end to show last frame
    }

    updateSeekBar();
  }

  // Event listeners
  btnPlayPause.addEventListener('click', handlePlayPause);
  seekBar.addEventListener('input', handleSeekBarInput);
  videoElement.addEventListener('timeupdate', handleTimeUpdate);
  videoElement.addEventListener('play', updatePlayPauseButton);
  videoElement.addEventListener('pause', updatePlayPauseButton);

  // Also allow clicking on video to play/pause
  videoElement.addEventListener('click', handlePlayPause);

  // Initialize displays
  updateSeekBar();
  updatePlayPauseButton();

  // Store reference to update function for trim slider integration
  window._updateVideoSeekBar = updateSeekBar;
}
