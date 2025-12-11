/**
 * Frame Selection Module
 * 
 * Handles key frame selection functionality.
 */

/**
 * Load frames for selection after extraction
 * @param {string} uploadId - Upload ID
 * @param {string} videoName - Video name
 */
async function loadFramesForSelection(uploadId, videoName) {
  try {
    const response = await fetch(`/api/workflow/frames/${uploadId}/${videoName}`);
    if (!response.ok) {
      throw new Error('Failed to load frames');
    }

    const data = await response.json();
    const frameState = WorkflowState.getFrameSelectionState();
    frameState.uploadId = uploadId;
    frameState.videoName = videoName;
    frameState.frames = data.frames;
    frameState.currentIndex = 0;
    frameState.selectedFrames = new Set(
      data.frames.filter(f => f.selected).map(f => f.filename)
    );

    // Update UI
    const frameTotal = document.getElementById('frame-total');
    const frameSlider = document.getElementById('frame-slider');

    if (frameTotal) {
      frameTotal.textContent = data.frames.length;
    }
    updateSelectedFramesCounter();
    if (frameSlider) {
      frameSlider.max = data.frames.length - 1;
    }

    // Load first frame
    WorkflowState.setFrameAspectRatio(null); // Reset aspect ratio detection
    updateFramePreview();

    // Update previously selected frames to show auto-selected first frame
    updatePreviouslySelectedFrames();

  } catch (error) {
    console.error('Failed to load frames:', error);
    showStatus('step-1', `Error loading frames: ${error.message}`, 'error');
  }
}

/**
 * Update frame preview
 */
function updateFramePreview() {
  const frameState = WorkflowState.getFrameSelectionState();
  const frame = frameState.frames[frameState.currentIndex];
  if (!frame) return;

  // Update current frame in both layouts
  const framePreview = document.getElementById('frame-preview');
  const currentHorizontalImg = document.getElementById('frame-preview-current-horizontal');

  if (framePreview) {
    framePreview.src = frame.path;
  }
  if (currentHorizontalImg) {
    currentHorizontalImg.src = frame.path;
  }

  const frameCurrent = document.getElementById('frame-current');
  const frameSlider = document.getElementById('frame-slider');

  if (frameCurrent) {
    frameCurrent.textContent = frameState.currentIndex + 1;
  }
  if (frameSlider) {
    frameSlider.value = frameState.currentIndex;
  }

  // Update selected badge (both vertical and horizontal layouts)
  const badge = document.getElementById('frame-selected-badge');
  const badgeHorizontal = document.getElementById('frame-selected-badge-horizontal');
  const isSelected = frameState.selectedFrames.has(frame.filename);
  if (badge) {
    badge.style.display = isSelected ? 'block' : 'none';
  }
  if (badgeHorizontal) {
    badgeHorizontal.style.display = isSelected ? 'block' : 'none';
  }

  // Update previously selected frames display
  updatePreviouslySelectedFrames();

  // Update frame counter display
  updateFrameCounterDisplay();

  // Determine aspect ratio from first frame
  if (WorkflowState.getFrameAspectRatio() === null) {
    detectAspectRatio(frame);
  }
}

/**
 * Detect aspect ratio from image
 * @param {Object} frame - Frame object
 */
function detectAspectRatio(frame) {
  const img = new Image();
  img.onload = () => {
    const aspectRatio = img.width >= img.height ? 'horizontal' : 'vertical';
    WorkflowState.setFrameAspectRatio(aspectRatio);
    console.log(`[Frame Aspect] Detected: ${aspectRatio} (${img.width}x${img.height})`);
    updateLayoutForAspectRatio();
  };
  img.src = frame.path;
}

/**
 * Update layout based on detected aspect ratio
 */
function updateLayoutForAspectRatio() {
  const verticalLayout = document.getElementById('vertical-layout');
  const horizontalLayout = document.getElementById('horizontal-layout');

  if (!verticalLayout || !horizontalLayout) return;

  const aspectRatio = WorkflowState.getFrameAspectRatio();
  if (aspectRatio === 'vertical') {
    verticalLayout.style.display = 'grid';
    horizontalLayout.style.display = 'none';
  } else {
    verticalLayout.style.display = 'none';
    horizontalLayout.style.display = 'block';
  }
}

/**
 * Update previously selected frames display
 */
function updatePreviouslySelectedFrames() {
  const frameState = WorkflowState.getFrameSelectionState();
  // Get the first selected frame
  const selectedFrames = frameState.frames.filter(f =>
    frameState.selectedFrames.has(f.filename)
  );

  const firstSelectedFrame = selectedFrames.length > 0 ? selectedFrames[0] : null;

  // Update both layouts regardless of aspect ratio
  const prevImgVertical = document.getElementById('frame-preview-prev-vertical');
  const prevImgHorizontal = document.getElementById('frame-preview-prev-horizontal');

  if (firstSelectedFrame) {
    if (prevImgVertical) prevImgVertical.src = firstSelectedFrame.path;
    if (prevImgHorizontal) prevImgHorizontal.src = firstSelectedFrame.path;
  } else {
    if (prevImgVertical) prevImgVertical.src = '';
    if (prevImgHorizontal) prevImgHorizontal.src = '';
  }
}

/**
 * Handle frame slider change
 * @param {Event} event - Slider change event
 */
function handleFrameSliderChange(event) {
  const sliderValue = parseInt(event.target.value);
  const frameState = WorkflowState.getFrameSelectionState();

  if (frameState.viewMode === 'selected') {
    // In selected mode, slider represents index in selected frames array
    const selectedFrames = getSelectedFramesArray();
    if (sliderValue >= 0 && sliderValue < selectedFrames.length) {
      const selectedFrame = selectedFrames[sliderValue];
      frameState.currentIndex = frameState.frames.findIndex(
        f => f.filename === selectedFrame.filename
      );
    }
  } else {
    // In all mode, slider represents index in all frames
    frameState.currentIndex = sliderValue;
  }

  updateFramePreview();
}

/**
 * Handle keyboard shortcuts
 * @param {KeyboardEvent} event - Keyboard event
 */
function handleKeyboardShortcuts(event) {
  // Only handle shortcuts when frame selection UI is visible
  const frameSelectionUI = document.getElementById('frame-selection-ui');
  if (!frameSelectionUI || frameSelectionUI.style.display === 'none') {
    return;
  }

  // Ignore if typing in input field
  if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') {
    return;
  }

  switch (event.key.toLowerCase()) {
    case 'arrowleft':
      event.preventDefault();
      navigateFrame(-1);
      break;
    case 'arrowright':
      event.preventDefault();
      navigateFrame(1);
      break;
    case 's':
      event.preventDefault();
      selectCurrentFrame();
      break;
    case 'x':
      event.preventDefault();
      deselectCurrentFrame();
      break;
  }
}

/**
 * Navigate to next/previous frame
 * @param {number} direction - Direction (-1 for previous, 1 for next)
 */
function navigateFrame(direction) {
  const frameState = WorkflowState.getFrameSelectionState();

  if (frameState.viewMode === 'selected') {
    // In selected mode, navigate only through selected frames
    const selectedFrames = getSelectedFramesArray();
    if (selectedFrames.length === 0) return;

    const currentFrame = frameState.frames[frameState.currentIndex];
    const currentSelectedIndex = selectedFrames.findIndex(f => f.filename === currentFrame.filename);

    const newSelectedIndex = currentSelectedIndex + direction;
    if (newSelectedIndex >= 0 && newSelectedIndex < selectedFrames.length) {
      const newFrame = selectedFrames[newSelectedIndex];
      frameState.currentIndex = frameState.frames.findIndex(f => f.filename === newFrame.filename);
      updateFramePreview();
      updateSliderForViewMode();
    }
  } else {
    // In all frames mode, navigate normally
    const newIndex = frameState.currentIndex + direction;
    if (newIndex >= 0 && newIndex < frameState.frames.length) {
      frameState.currentIndex = newIndex;
      updateFramePreview();
    }
  }
}

/**
 * Select current frame
 */
async function selectCurrentFrame() {
  const frameState = WorkflowState.getFrameSelectionState();
  const frame = frameState.frames[frameState.currentIndex];
  if (!frame || frameState.selectedFrames.has(frame.filename)) {
    return; // Already selected
  }

  try {
    const response = await fetch(
      `/api/workflow/frames/${frameState.uploadId}/${frameState.videoName}/select?frame_name=${frame.filename}`,
      { method: 'POST' }
    );

    if (!response.ok) {
      throw new Error('Failed to select frame');
    }

    frameState.selectedFrames.add(frame.filename);
    updateSelectedFramesCounter();
    updateFramePreview();

  } catch (error) {
    console.error('Failed to select frame:', error);
  }
}

/**
 * Deselect current frame
 */
async function deselectCurrentFrame() {
  const frameState = WorkflowState.getFrameSelectionState();
  const frame = frameState.frames[frameState.currentIndex];
  if (!frame || !frameState.selectedFrames.has(frame.filename)) {
    return; // Not selected
  }

  try {
    const response = await fetch(
      `/api/workflow/frames/${frameState.uploadId}/${frameState.videoName}/select/${frame.filename}`,
      { method: 'DELETE' }
    );

    if (!response.ok) {
      throw new Error('Failed to deselect frame');
    }

    frameState.selectedFrames.delete(frame.filename);
    updateSelectedFramesCounter();
    updateFramePreview();

  } catch (error) {
    console.error('Failed to deselect frame:', error);
  }
}

/**
 * Save selected frames to training pool
 */
async function saveToTrainingPool() {
  const frameState = WorkflowState.getFrameSelectionState();

  if (frameState.selectedFrames.size === 0) {
    alert('Please select at least one frame before saving');
    return;
  }

  if (!confirm(`Save ${frameState.selectedFrames.size} selected frames to training pool?`)) {
    return;
  }

  try {
    showStatus('step-1', 'Saving to training pool...', 'info');

    const response = await fetch('/api/workflow/save-to-training-pool', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        upload_id: frameState.uploadId,
        video_name: frameState.videoName,
      }),
    });

    if (!response.ok) {
      throw new Error('Failed to save to training pool');
    }

    const data = await response.json();

    showStatus('step-1', `Saved to training pool! Total: ${data.total_videos} videos, ${data.total_frames} frames`, 'success');
    alert(`✓ Saved to training pool!\n\nTotal videos: ${data.total_videos}\nTotal frames: ${data.total_frames}`);

    // Update pool info display
    if (typeof loadTrainingPoolInfo === 'function') {
      loadTrainingPoolInfo();
    }

  } catch (error) {
    console.error('Failed to save to pool:', error);
    showStatus('step-1', `Save failed: ${error.message}`, 'error');
  }
}

/**
 * Train frame selector model (current video only)
 */
async function trainFrameSelector() {
  const frameState = WorkflowState.getFrameSelectionState();

  if (frameState.selectedFrames.size === 0) {
    alert('Please select at least one frame before training');
    return;
  }

  if (!confirm(`Train frame selector model with ${frameState.selectedFrames.size} selected frames from current video only?`)) {
    return;
  }

  try {
    showStatus('step-1', 'Training frame selector model...', 'info');

    const response = await fetch('/api/workflow/train-frame-selector', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        upload_id: frameState.uploadId,
        video_name: frameState.videoName,
      }),
    });

    if (!response.ok) {
      throw new Error('Training failed');
    }

    const data = await response.json();

    if (data.note) {
      // Training pipeline not yet implemented
      showStatus('step-1', `${data.message} (${data.note})`, 'info');
      alert(`✓ ${data.message}\n\nNote: ${data.note}`);
    } else {
      showStatus('step-1', `Training complete!`, 'success');
      alert(`✓ Training complete!\n\nTest F1 Score: ${(data.results?.metrics?.f1 * 100 || 0).toFixed(1)}%`);
    }

  } catch (error) {
    console.error('Training failed:', error);
    showStatus('step-1', `Training failed: ${error.message}`, 'error');
  }
}

/**
 * Clear all data (uploads and workflow_frames)
 */
async function clearAllData() {
  if (!confirm('⚠️ This will delete ALL uploads and workflow data. Continue?')) {
    return;
  }

  if (!confirm('🚨 Are you REALLY sure? This cannot be undone!')) {
    return;
  }

  try {
    const response = await fetch('/api/system/clear', {
      method: 'POST',
    });

    if (!response.ok) {
      throw new Error('Failed to clear data');
    }

    const data = await response.json();
    alert(`✓ Data cleared successfully!\nCleared: ${data.cleared_directories.join(', ')}`);

    // Reset UI state
    WorkflowState.reset();

    // Reload page
    location.reload();

  } catch (error) {
    console.error('Failed to clear data:', error);
    alert(`❌ Error: ${error.message}`);
  }
}

// ========== View Mode Helpers ==========

/**
 * Get array of selected frames in order
 * @returns {Array} Selected frames array
 */
function getSelectedFramesArray() {
  const frameState = WorkflowState.getFrameSelectionState();
  return frameState.frames.filter(f =>
    frameState.selectedFrames.has(f.filename)
  );
}

/**
 * Update selected frames counter display
 */
function updateSelectedFramesCounter() {
  const frameState = WorkflowState.getFrameSelectionState();
  const count = frameState.selectedFrames.size;

  const frameSelectedCount = document.getElementById('frame-selected-count');
  const selectedFramesCounter = document.getElementById('selected-frames-counter');

  if (frameSelectedCount) {
    frameSelectedCount.textContent = count;
  }
  if (selectedFramesCounter) {
    selectedFramesCounter.textContent = `${count} selected`;
  }
}

/**
 * Set view mode (all or selected)
 * @param {string} mode - View mode ('all' or 'selected')
 */
function setViewMode(mode) {
  const frameState = WorkflowState.getFrameSelectionState();
  frameState.viewMode = mode;

  // Update button styles
  const btnAll = document.getElementById('btn-view-all');
  const btnSelected = document.getElementById('btn-view-selected');

  if (mode === 'all') {
    if (btnAll) {
      btnAll.style.background = '#0066cc';
      btnAll.style.color = 'white';
    }
    if (btnSelected) {
      btnSelected.style.background = '#333';
      btnSelected.style.color = '#aaa';
    }
  } else {
    if (btnAll) {
      btnAll.style.background = '#333';
      btnAll.style.color = '#aaa';
    }
    if (btnSelected) {
      btnSelected.style.background = '#ff9900';
      btnSelected.style.color = 'white';
    }

    // If no frames selected, show alert and stay in all mode
    if (frameState.selectedFrames.size === 0) {
      alert('No frames selected yet. Please select at least one frame.');
      setViewMode('all');
      return;
    }

    // Jump to first selected frame
    const selectedFrames = getSelectedFramesArray();
    if (selectedFrames.length > 0) {
      frameState.currentIndex = frameState.frames.findIndex(
        f => f.filename === selectedFrames[0].filename
      );
    }
  }

  updateFramePreview();
  updateSliderForViewMode();
}

/**
 * Update slider for current view mode
 */
function updateSliderForViewMode() {
  const slider = document.getElementById('frame-slider');
  if (!slider) return;

  const frameState = WorkflowState.getFrameSelectionState();

  if (frameState.viewMode === 'selected') {
    const selectedFrames = getSelectedFramesArray();
    const currentFrame = frameState.frames[frameState.currentIndex];
    const selectedIndex = selectedFrames.findIndex(f => f.filename === currentFrame.filename);

    slider.max = Math.max(0, selectedFrames.length - 1);
    slider.value = selectedIndex;
  } else {
    slider.max = Math.max(0, frameState.frames.length - 1);
    slider.value = frameState.currentIndex;
  }
}

/**
 * Update frame counter display (e.g., "3 / 120" or "2 / 5 selected")
 */
function updateFrameCounterDisplay() {
  const currentElem = document.getElementById('frame-current');
  const totalElem = document.getElementById('frame-total');
  if (!currentElem || !totalElem) return;

  const frameState = WorkflowState.getFrameSelectionState();

  if (frameState.viewMode === 'selected') {
    const selectedFrames = getSelectedFramesArray();
    const currentFrame = frameState.frames[frameState.currentIndex];
    const selectedIndex = selectedFrames.findIndex(f => f.filename === currentFrame.filename);

    currentElem.textContent = selectedIndex + 1;
    totalElem.textContent = selectedFrames.length;
  } else {
    currentElem.textContent = frameState.currentIndex + 1;
    totalElem.textContent = frameState.frames.length;
  }
}
