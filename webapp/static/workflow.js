/**
 * Hold Labeling Workflow - Frontend JavaScript
 * 
 * Manages the end-to-end workflow from video upload to model deployment.
 */

// State
let currentFrameDir = null;
let currentSessionId = null;
let currentTrainingJobId = null;
let frameAspectRatio = null; // 'vertical' or 'horizontal'
let currentUploadId = null;
let currentVideoName = null;
let holdColor = 'red';
let routeDifficulty = 'beginner';
let firstFrameImageUrl = null; // Store first frame for both preview and hold detection
let threeViewer = null; // Three.js viewer instance
let current3DModelId = null; // Current 3D model ID

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  setupEventListeners();
  loadSessions();
  loadTrainingJobs();

  // Initially hide step 2, 4 - they will be shown after "Start Analyzing"
  // Step 3 (frame selection) is shown for demo purposes
  const frameSelectionUI = document.getElementById('frame-selection-ui');
  const step2 = document.getElementById('step-2');
  const step3 = document.getElementById('step-3');
  const step4 = document.getElementById('step-4');

  // Demo: Show step 3 (frame selection) and its UI immediately
  if (step3) step3.style.display = 'block';
  if (frameSelectionUI) frameSelectionUI.style.display = 'block';
  if (step2) step2.style.display = 'none';
  if (step4) step4.style.display = 'none';

  // Poll for updates every 5 seconds
  setInterval(() => {
    loadSessions();
    loadTrainingJobs();
    // Load training pools if visible
    const poolHoldsContent = document.getElementById('pool-holds-content');
    const poolFramesContent = document.getElementById('pool-frames-content');
    if (poolHoldsContent && poolHoldsContent.style.display !== 'none') {
      loadHoldDetectionPool();
    }
    if (poolFramesContent && poolFramesContent.style.display !== 'none') {
      loadKeyFrameSelectionPool();
    }
  }, 5000);

  // Load initial training pool (holds by default)
  loadHoldDetectionPool();
});

/**
 * Setup event listeners
 */
function setupEventListeners() {
  // Video file selection - show first frame preview
  const videoFileInput = document.getElementById('video-file');
  if (videoFileInput) {
    videoFileInput.addEventListener('change', function (event) {
      handleVideoFileSelection(event);
    });
  }

  // 3D conversion button (temporarily disabled)
  // const btnConvert3D = document.getElementById('btn-convert-to-3d');
  // if (btnConvert3D) {
  //   btnConvert3D.addEventListener('click', handleConvertTo3D);
  // }

  // Close 3D viewer button (temporarily disabled)
  // const btnClose3DViewer = document.getElementById('btn-close-3d-viewer');
  // if (btnClose3DViewer) {
  //   btnClose3DViewer.addEventListener('click', close3DViewer);
  // }

  // Hold color and route difficulty dropdown changes
  const holdColorSelect = document.getElementById('hold-color');
  if (holdColorSelect) {
    holdColorSelect.addEventListener('change', updateVideoPreview);
  }

  const routeDifficultySelect = document.getElementById('route-difficulty');
  if (routeDifficultySelect) {
    routeDifficultySelect.addEventListener('change', updateVideoPreview);
  }

  // Other event listeners
  const btnExtractFrames = document.getElementById('btn-extract-frames');
  if (btnExtractFrames) {
    btnExtractFrames.addEventListener('click', startAnalyzing);
  }

  const btnCreateSession = document.getElementById('btn-create-session');
  if (btnCreateSession) {
    btnCreateSession.addEventListener('click', createSession);
  }

  const btnStartTraining = document.getElementById('btn-start-training');
  if (btnStartTraining) {
    btnStartTraining.addEventListener('click', startTraining);
  }

  const btnUploadGCS = document.getElementById('btn-upload-gcs');
  if (btnUploadGCS) {
    btnUploadGCS.addEventListener('click', uploadToGCS);
  }

  const btnClearData = document.getElementById('btn-clear-data');
  if (btnClearData) {
    btnClearData.addEventListener('click', clearAllData);
  }

  // Frame selection UI
  const frameSlider = document.getElementById('frame-slider');
  if (frameSlider) {
    frameSlider.addEventListener('input', handleFrameSliderChange);
  }

  const btnSaveToPool = document.getElementById('btn-save-to-pool');
  if (btnSaveToPool) {
    btnSaveToPool.addEventListener('click', saveToTrainingPool);
  }

  const btnTrainFrameSelector = document.getElementById('btn-train-frame-selector');
  if (btnTrainFrameSelector) {
    btnTrainFrameSelector.addEventListener('click', trainFrameSelector);
  }

  const btnViewAll = document.getElementById('btn-view-all');
  if (btnViewAll) {
    btnViewAll.addEventListener('click', () => setViewMode('all'));
  }

  const btnViewSelected = document.getElementById('btn-view-selected');
  if (btnViewSelected) {
    btnViewSelected.addEventListener('click', () => setViewMode('selected'));
  }

  // Keyboard shortcuts for frame selection
  document.addEventListener('keydown', handleKeyboardShortcuts);

  // Load training pool info on page load
  loadTrainingPoolInfo();

  // Hold labeling submit button
  const btnSubmitHolds = document.getElementById('btn-submit-holds');
  if (btnSubmitHolds) {
    btnSubmitHolds.addEventListener('click', submitHoldLabels);
  }

  // Training pool toggle buttons
  const btnPoolToggleHolds = document.getElementById('btn-pool-toggle-holds');
  const btnPoolToggleFrames = document.getElementById('btn-pool-toggle-frames');
  if (btnPoolToggleHolds) {
    btnPoolToggleHolds.addEventListener('click', () => toggleTrainingPool('holds'));
  }
  if (btnPoolToggleFrames) {
    btnPoolToggleFrames.addEventListener('click', () => toggleTrainingPool('frames'));
  }

  // Training pool train buttons
  const btnTrainYoloFromPool = document.getElementById('btn-train-yolo-from-pool');
  const btnTrainFrameSelectorFromPool = document.getElementById('btn-train-frame-selector-from-pool');
  if (btnTrainYoloFromPool) {
    btnTrainYoloFromPool.addEventListener('click', trainYoloFromPool);
  }
  if (btnTrainFrameSelectorFromPool) {
    btnTrainFrameSelectorFromPool.addEventListener('click', trainFrameSelectorFromPool);
  }
}

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
    currentFrameDir = data.frame_directory;

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
    currentUploadId = uploadId;
    currentVideoName = videoName;
    currentFrameDir = data.frame_directory;

    showStatus('step-1', `Extracted ${data.frame_count} frames`, 'success');
    setStepCompleted('step-1');

    // Show Step 2 and hold labeling UI
    const step2 = document.getElementById('step-2');
    if (step2) {
      step2.style.display = 'block';
      setStepActive('step-2');
    }

    // Show hold labeling UI and automatically start SAM segmentation
    showHoldLabelingUI(uploadId, videoName, data.frame_count);

  } catch (error) {
    console.error('Frame extraction failed:', error);
    showStatus('step-1', `Error: ${error.message}`, 'error');
  }
}

/**
 * Step 2: Create labeling session
 */
async function createSession() {
  const sessionName = document.getElementById('session-name').value.trim();
  if (!sessionName) {
    showStatus('step-2', 'Please enter a session name', 'error');
    return;
  }

  if (!currentFrameDir) {
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
        frame_dir: currentFrameDir,
        use_sam: true,  // Enable SAM segmentation
        sam_checkpoint: null,  // Will use default checkpoint
      }),
    });

    if (!response.ok) {
      throw new Error(`Failed to create session: ${response.statusText}`);
    }

    const data = await response.json();
    currentSessionId = data.session_id;

    showStatus('step-2', 'Session created! Opening labeling UI...', 'success');

    // Open labeling UI in new tab
    setTimeout(() => {
      window.open(`/labeling?session_id=${currentSessionId}`, '_blank');
    }, 1000);

    setStepCompleted('step-2');
    loadSessions();

  } catch (error) {
    console.error('Session creation failed:', error);
    showStatus('step-2', `Error: ${error.message}`, 'error');
  }
}

/**
 * Step 3: Start YOLO training
 */
async function startTraining() {
  if (!currentSessionId) {
    showStatus('step-3', 'No labeling session selected', 'error');
    return;
  }

  const epochs = parseInt(document.getElementById('train-epochs').value);
  const batch = parseInt(document.getElementById('train-batch').value);

  showStatus('step-3', 'Starting YOLO training...', 'info');

  try {
    // First, export the session to YOLO format
    const exportResponse = await fetch(`/api/labeling/sessions/${currentSessionId}/export`, {
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
    currentTrainingJobId = trainData.job_id;

    showStatus('step-3', `Training started (Job ID: ${currentTrainingJobId})`, 'success');
    setStepCompleted('step-3');
    setStepActive('step-4');

    document.getElementById('btn-upload-gcs').disabled = true; // Enable after training completes
    loadTrainingJobs();

  } catch (error) {
    console.error('Training failed:', error);
    showStatus('step-3', `Error: ${error.message}`, 'error');
  }
}

/**
 * Step 4: Upload model to GCS
 */
async function uploadToGCS() {
  if (!currentTrainingJobId) {
    showStatus('step-4', 'No training job selected', 'error');
    return;
  }

  showStatus('step-4', 'Uploading model to GCS...', 'info');

  try {
    const response = await fetch(`/api/yolo/train/jobs/${currentTrainingJobId}/upload`, {
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

/**
 * Load and display labeling sessions
 */
async function loadSessions() {
  try {
    const response = await fetch('/api/labeling/sessions');
    if (!response.ok) return;

    const data = await response.json();
    const container = document.getElementById('sessions-container');

    if (data.sessions.length === 0) {
      container.innerHTML = '<p style="color: #888;">No sessions yet. Start segmentation to create one.</p>';
      return;
    }

    container.innerHTML = data.sessions.map(session => `
            <div class="session-item">
                <div class="session-info">
                    <div class="session-name">${session.name}</div>
                    <div class="session-meta">
                        Status: ${session.status} | 
                        Frames: ${session.frame_count} | 
                        Labeled: ${session.labeled_segments} holds
                    </div>
                </div>
                <div class="session-actions">
                    <button onclick="openSession('${session.id}')">Open</button>
                    <button onclick="exportSession('${session.id}')">Export</button>
                    <button onclick="selectSession('${session.id}')">Use for Training</button>
                </div>
            </div>
        `).join('');

  } catch (error) {
    console.error('Failed to load sessions:', error);
  }
}

/**
 * Load and display training jobs
 */
async function loadTrainingJobs() {
  try {
    const response = await fetch('/api/yolo/train/jobs');
    if (!response.ok) return;

    const data = await response.json();
    const container = document.getElementById('jobs-container');

    if (data.jobs.length === 0) {
      container.innerHTML = '<p style="color: #888;">No training jobs yet. Train after labeling/selection.</p>';
      return;
    }

    container.innerHTML = data.jobs.map(job => `
            <div class="training-job">
                <div class="job-header">
                    <div class="job-id">Job ID: ${job.id}</div>
                    <div class="job-status ${job.status}">${job.status.toUpperCase()}</div>
                </div>
                <div style="color: #aaa; font-size: 12px; margin-bottom: 8px;">
                    Epochs: ${job.epochs} | Batch: ${job.batch} | Model: ${job.model}
                </div>
                ${job.status === 'completed' ? `
                    <div style="color: #00cc66; font-size: 12px; margin-bottom: 8px;">
                        Model: ${job.best_model_path}
                    </div>
                    ${job.gcs_uri ? `
                        <div style="color: #0066cc; font-size: 11px;">
                            GCS: ${job.gcs_uri}
                        </div>
                    ` : `
                        <button onclick="uploadJobToGCS('${job.id}')" style="padding: 6px 12px; margin-top: 8px;">
                            Upload to GCS
                        </button>
                    `}
                ` : ''}
            </div>
        `).join('');

    // Enable GCS upload button if training completed
    const completedJobs = data.jobs.filter(j => j.status === 'completed');
    if (completedJobs.length > 0 && currentTrainingJobId) {
      const currentJob = data.jobs.find(j => j.id === currentTrainingJobId);
      if (currentJob && currentJob.status === 'completed') {
        document.getElementById('btn-upload-gcs').disabled = false;
      }
    }

  } catch (error) {
    console.error('Failed to load training jobs:', error);
  }
}

/**
 * Helper: Open labeling session
 */
function openSession(sessionId) {
  window.open(`/labeling?session_id=${sessionId}`, '_blank');
}

/**
 * Helper: Export session
 */
async function exportSession(sessionId) {
  try {
    const response = await fetch(`/api/labeling/sessions/${sessionId}/export`, {
      method: 'POST',
    });

    if (!response.ok) {
      throw new Error(`Export failed: ${response.statusText}`);
    }

    const data = await response.json();
    alert(`Exported ${data.exported_count} images to YOLO dataset`);

  } catch (error) {
    alert(`Error: ${error.message}`);
  }
}

/**
 * Helper: Select session for training
 */
function selectSession(sessionId) {
  currentSessionId = sessionId;
  document.getElementById('btn-start-training').disabled = false;
  setStepActive('step-3');
  alert(`Session selected. You can now start training in Step 3.`);
}

/**
 * Helper: Upload job model to GCS
 */
async function uploadJobToGCS(jobId) {
  try {
    const response = await fetch(`/api/yolo/train/jobs/${jobId}/upload`, {
      method: 'POST',
    });

    if (!response.ok) {
      throw new Error(`Upload failed: ${response.statusText}`);
    }

    const data = await response.json();
    alert(`Model uploaded to: ${data.gcs_uri}`);
    loadTrainingJobs();

  } catch (error) {
    alert(`Error: ${error.message}`);
  }
}

/**
 * Helper: Show status message
 */
function showStatus(stepId, message, type) {
  const el = document.getElementById(`status-${stepId}`);
  el.textContent = message;
  el.className = `step-status ${type}`;
  el.style.display = 'block';
}

/**
 * Helper: Set step as active
 */
function setStepActive(stepId) {
  document.querySelectorAll('.step-card').forEach(card => {
    card.classList.remove('active');
  });
  document.getElementById(stepId).classList.add('active');
}

/**
 * Helper: Set step as completed
 */
function setStepCompleted(stepId) {
  document.getElementById(stepId).classList.add('completed');
}

// ========== Frame Selection Learning ==========

let frameSelectionState = {
  uploadId: null,
  videoName: null,
  frames: [],
  currentIndex: 0,
  selectedFrames: new Set(),
  viewMode: 'all', // 'all' or 'selected'
};

// Pipeline mode removed: unified flow now handles hold labeling then keyframe selection


/**
 * Load frames for selection after extraction
 */
async function loadFramesForSelection(uploadId, videoName) {
  try {
    const response = await fetch(`/api/workflow/frames/${uploadId}/${videoName}`);
    if (!response.ok) {
      throw new Error('Failed to load frames');
    }

    const data = await response.json();
    frameSelectionState.uploadId = uploadId;
    frameSelectionState.videoName = videoName;
    frameSelectionState.frames = data.frames;
    frameSelectionState.currentIndex = 0;
    frameSelectionState.selectedFrames = new Set(
      data.frames.filter(f => f.selected).map(f => f.filename)
    );

    // Update UI
    document.getElementById('frame-total').textContent = data.frames.length;
    updateSelectedFramesCounter();
    document.getElementById('frame-slider').max = data.frames.length - 1;

    // Load first frame
    frameAspectRatio = null; // Reset aspect ratio detection
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
  const frame = frameSelectionState.frames[frameSelectionState.currentIndex];
  if (!frame) return;

  // Update current frame in both layouts
  document.getElementById('frame-preview').src = frame.path;
  const currentHorizontalImg = document.getElementById('frame-preview-current-horizontal');
  if (currentHorizontalImg) {
    currentHorizontalImg.src = frame.path;
  }

  document.getElementById('frame-current').textContent = frameSelectionState.currentIndex + 1;
  document.getElementById('frame-slider').value = frameSelectionState.currentIndex;

  // Update selected badge (both vertical and horizontal layouts)
  const badge = document.getElementById('frame-selected-badge');
  const badgeHorizontal = document.getElementById('frame-selected-badge-horizontal');
  const isSelected = frameSelectionState.selectedFrames.has(frame.filename);
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
  if (frameAspectRatio === null) {
    detectAspectRatio(frame);
  }
}

/**
 * Detect aspect ratio from image
 */
function detectAspectRatio(frame) {
  const img = new Image();
  img.onload = () => {
    frameAspectRatio = img.width >= img.height ? 'horizontal' : 'vertical';
    console.log(`[Frame Aspect] Detected: ${frameAspectRatio} (${img.width}x${img.height})`);
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

  if (frameAspectRatio === 'vertical') {
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
  // Get the first selected frame
  const selectedFrames = frameSelectionState.frames.filter(f =>
    frameSelectionState.selectedFrames.has(f.filename)
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
}/**
 * Handle frame slider change
 */
function handleFrameSliderChange(event) {
  const sliderValue = parseInt(event.target.value);

  if (frameSelectionState.viewMode === 'selected') {
    // In selected mode, slider represents index in selected frames array
    const selectedFrames = getSelectedFramesArray();
    if (sliderValue >= 0 && sliderValue < selectedFrames.length) {
      const selectedFrame = selectedFrames[sliderValue];
      frameSelectionState.currentIndex = frameSelectionState.frames.findIndex(
        f => f.filename === selectedFrame.filename
      );
    }
  } else {
    // In all mode, slider represents index in all frames
    frameSelectionState.currentIndex = sliderValue;
  }

  updateFramePreview();
}

/**
 * Handle keyboard shortcuts
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
 */
function navigateFrame(direction) {
  if (frameSelectionState.viewMode === 'selected') {
    // In selected mode, navigate only through selected frames
    const selectedFrames = getSelectedFramesArray();
    if (selectedFrames.length === 0) return;

    const currentFrame = frameSelectionState.frames[frameSelectionState.currentIndex];
    const currentSelectedIndex = selectedFrames.findIndex(f => f.filename === currentFrame.filename);

    const newSelectedIndex = currentSelectedIndex + direction;
    if (newSelectedIndex >= 0 && newSelectedIndex < selectedFrames.length) {
      const newFrame = selectedFrames[newSelectedIndex];
      frameSelectionState.currentIndex = frameSelectionState.frames.findIndex(f => f.filename === newFrame.filename);
      updateFramePreview();
      updateSliderForViewMode();
    }
  } else {
    // In all frames mode, navigate normally
    const newIndex = frameSelectionState.currentIndex + direction;
    if (newIndex >= 0 && newIndex < frameSelectionState.frames.length) {
      frameSelectionState.currentIndex = newIndex;
      updateFramePreview();
    }
  }
}

/**
 * Select current frame
 */
async function selectCurrentFrame() {
  const frame = frameSelectionState.frames[frameSelectionState.currentIndex];
  if (!frame || frameSelectionState.selectedFrames.has(frame.filename)) {
    return; // Already selected
  }

  try {
    const response = await fetch(
      `/api/workflow/frames/${frameSelectionState.uploadId}/${frameSelectionState.videoName}/select?frame_name=${frame.filename}`,
      { method: 'POST' }
    );

    if (!response.ok) {
      throw new Error('Failed to select frame');
    }

    frameSelectionState.selectedFrames.add(frame.filename);
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
  const frame = frameSelectionState.frames[frameSelectionState.currentIndex];
  if (!frame || !frameSelectionState.selectedFrames.has(frame.filename)) {
    return; // Not selected
  }

  try {
    const response = await fetch(
      `/api/workflow/frames/${frameSelectionState.uploadId}/${frameSelectionState.videoName}/select/${frame.filename}`,
      { method: 'DELETE' }
    );

    if (!response.ok) {
      throw new Error('Failed to deselect frame');
    }

    frameSelectionState.selectedFrames.delete(frame.filename);
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
  if (frameSelectionState.selectedFrames.size === 0) {
    alert('Please select at least one frame before saving');
    return;
  }

  if (!confirm(`Save ${frameSelectionState.selectedFrames.size} selected frames to training pool?`)) {
    return;
  }

  try {
    showStatus('step-1', 'Saving to training pool...', 'info');

    const response = await fetch('/api/workflow/save-to-training-pool', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        upload_id: frameSelectionState.uploadId,
        video_name: frameSelectionState.videoName,
      }),
    });

    if (!response.ok) {
      throw new Error('Failed to save to training pool');
    }

    const data = await response.json();

    showStatus('step-1', `Saved to training pool! Total: ${data.total_videos} videos, ${data.total_frames} frames`, 'success');
    alert(`✓ Saved to training pool!\n\nTotal videos: ${data.total_videos}\nTotal frames: ${data.total_frames}`);

    // Update pool info display
    loadTrainingPoolInfo();

  } catch (error) {
    console.error('Failed to save to pool:', error);
    showStatus('step-1', `Save failed: ${error.message}`, 'error');
  }
}

/**
 * Load training pool info
 */
async function loadTrainingPoolInfo() {
  try {
    const response = await fetch('/api/workflow/training-pool-info');
    if (!response.ok) {
      console.warn('Failed to load training pool info');
      return;
    }

    const data = await response.json();

    document.getElementById('pool-video-count').textContent = data.video_count || 0;
    document.getElementById('pool-frame-count').textContent = data.frame_count || 0;

  } catch (error) {
    console.error('Failed to load pool info:', error);
  }
}

/**
 * Train frame selector model (current video only)
 */
async function trainFrameSelector() {
  if (frameSelectionState.selectedFrames.size === 0) {
    alert('Please select at least one frame before training');
    return;
  }

  if (!confirm(`Train frame selector model with ${frameSelectionState.selectedFrames.size} selected frames from current video only?`)) {
    return;
  }

  try {
    showStatus('step-1', 'Training frame selector model...', 'info');

    const response = await fetch('/api/workflow/train-frame-selector', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        upload_id: frameSelectionState.uploadId,
        video_name: frameSelectionState.videoName,
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
    currentFrameDir = null;
    currentSessionId = null;
    currentTrainingJobId = null;
    frameSelectionState = {
      uploadId: null,
      videoName: null,
      frames: [],
      currentIndex: 0,
      selectedFrames: new Set(),
      viewMode: 'all',
    };

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
 */
function getSelectedFramesArray() {
  return frameSelectionState.frames.filter(f =>
    frameSelectionState.selectedFrames.has(f.filename)
  );
}

/**
 * Update selected frames counter display
 */
function updateSelectedFramesCounter() {
  const count = frameSelectionState.selectedFrames.size;
  document.getElementById('frame-selected-count').textContent = count;
  document.getElementById('selected-frames-counter').textContent =
    `${count} selected`;
}

/**
 * Set view mode (all or selected)
 */
function setViewMode(mode) {
  frameSelectionState.viewMode = mode;

  // Update button styles
  const btnAll = document.getElementById('btn-view-all');
  const btnSelected = document.getElementById('btn-view-selected');

  if (mode === 'all') {
    btnAll.style.background = '#0066cc';
    btnAll.style.color = 'white';
    btnSelected.style.background = '#333';
    btnSelected.style.color = '#aaa';
  } else {
    btnAll.style.background = '#333';
    btnAll.style.color = '#aaa';
    btnSelected.style.background = '#ff9900';
    btnSelected.style.color = 'white';

    // If no frames selected, show alert and stay in all mode
    if (frameSelectionState.selectedFrames.size === 0) {
      alert('No frames selected yet. Please select at least one frame.');
      setViewMode('all');
      return;
    }

    // Jump to first selected frame
    const selectedFrames = getSelectedFramesArray();
    if (selectedFrames.length > 0) {
      frameSelectionState.currentIndex = frameSelectionState.frames.findIndex(
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

  if (frameSelectionState.viewMode === 'selected') {
    const selectedFrames = getSelectedFramesArray();
    const currentFrame = frameSelectionState.frames[frameSelectionState.currentIndex];
    const selectedIndex = selectedFrames.findIndex(f => f.filename === currentFrame.filename);

    slider.max = Math.max(0, selectedFrames.length - 1);
    slider.value = selectedIndex;
  } else {
    slider.max = Math.max(0, frameSelectionState.frames.length - 1);
    slider.value = frameSelectionState.currentIndex;
  }
}

/**
 * Update frame counter display (e.g., "3 / 120" or "2 / 5 selected")
 */
function updateFrameCounterDisplay() {
  const currentElem = document.getElementById('frame-current');
  const totalElem = document.getElementById('frame-total');

  if (frameSelectionState.viewMode === 'selected') {
    const selectedFrames = getSelectedFramesArray();
    const currentFrame = frameSelectionState.frames[frameSelectionState.currentIndex];
    const selectedIndex = selectedFrames.findIndex(f => f.filename === currentFrame.filename);

    currentElem.textContent = selectedIndex + 1;
    totalElem.textContent = selectedFrames.length;
  } else {
    currentElem.textContent = frameSelectionState.currentIndex + 1;
    totalElem.textContent = frameSelectionState.frames.length;
  }
}

// ========== Hold Labeling UI ==========

// State for hold labeling
let holdLabelingSegments = [];
let holdLabelingCanvas = null;
let holdLabelingCtx = null;
let holdLabelingImage = null;

/**
 * Show hold labeling UI after frame extraction in hold detection mode
 */
function showHoldLabelingUI(uploadId, videoName, frameCount) {
  // Get hold color and difficulty from selectors
  holdColor = document.getElementById('hold-color').value;
  routeDifficulty = document.getElementById('route-difficulty').value;

  // Show the hold labeling UI
  const holdLabelingUI = document.getElementById('hold-labeling-ui');
  holdLabelingUI.style.display = 'block';

  // Initialize canvas
  holdLabelingCanvas = document.getElementById('hold-labeling-canvas');
  holdLabelingCtx = holdLabelingCanvas.getContext('2d');

  // Load first frame for labeling and automatically start SAM segmentation
  loadFirstFrameForLabeling(uploadId, videoName);
}

/**
 * Load first frame for hold labeling and start SAM segmentation automatically
 */
async function loadFirstFrameForLabeling(uploadId, videoName) {
  try {
    const response = await fetch(`/api/workflow/frames/${uploadId}/${videoName}`);
    if (!response.ok) {
      throw new Error('Failed to load frames');
    }

    const data = await response.json();
    if (data.frames && data.frames.length > 0) {
      // Get first frame
      const firstFrame = data.frames[0];

      // Load image onto canvas
      await loadImageToCanvas(firstFrame.path);

      // Automatically start SAM segmentation
      await startSamSegmentation(uploadId, videoName, firstFrame.filename);
    }
  } catch (error) {
    console.error('Failed to load frame for labeling:', error);
    showStatus('step-2', `Error loading frames: ${error.message}`, 'error');
  }
}

/**
 * Load image onto canvas
 */
function loadImageToCanvas(imageUrl) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      holdLabelingImage = img;

      // Get container dimensions (the inner div that holds the canvas)
      const container = holdLabelingCanvas.parentElement; // Get the inner container div
      const containerWidth = container.clientWidth;
      const maxHeight = window.innerHeight * 0.8; // Max height to prevent too large

      // Calculate aspect ratio
      const imgAspect = img.width / img.height;
      const containerAspect = containerWidth / maxHeight;

      // Fit image to container while maintaining aspect ratio (no cropping)
      let displayWidth, displayHeight;
      if (imgAspect > containerAspect) {
        // Image is wider - fit to width
        displayWidth = containerWidth;
        displayHeight = displayWidth / imgAspect;
      } else {
        // Image is taller - fit to height
        displayHeight = Math.min(maxHeight, img.height);
        displayWidth = displayHeight * imgAspect;
      }

      // Set canvas display size to fit without cropping
      holdLabelingCanvas.style.width = `${displayWidth}px`;
      holdLabelingCanvas.style.height = `${displayHeight}px`;
      holdLabelingCanvas.style.maxWidth = '100%';
      holdLabelingCanvas.style.maxHeight = `${maxHeight}px`;

      // Set canvas internal size to match image (for drawing)
      holdLabelingCanvas.width = img.width;
      holdLabelingCanvas.height = img.height;

      // Draw image
      holdLabelingCtx.drawImage(img, 0, 0);
      resolve();
    };
    img.onerror = reject;
    img.src = imageUrl;
  });
}

/**
 * Start SAM segmentation on first frame with real-time progress
 */
async function startSamSegmentation(uploadId, videoName, frameFilename) {
  const loadingDiv = document.getElementById('sam-loading');
  const segmentsList = document.getElementById('segments-list');

  try {
    // Show loading state with progress indicator
    loadingDiv.style.display = 'block';
    loadingDiv.innerHTML = `
      <div style="text-align: center;">
        <div style="color: #fff; font-size: 16px; margin-bottom: 10px;">Running SAM segmentation...</div>
        <div style="background: #2a2a2a; border-radius: 4px; height: 8px; overflow: hidden; margin-bottom: 10px;">
          <div id="sam-progress-bar" style="background: #0066cc; height: 100%; width: 0%; transition: width 0.3s;"></div>
        </div>
        <div id="sam-progress-message" style="color: #aaa; font-size: 12px;">Initializing...</div>
      </div>
    `;

    // Use EventSource for real-time progress updates
    const eventSource = new EventSource(
      `/api/workflow/segment-first-frame-stream?` +
      `upload_id=${encodeURIComponent(uploadId)}&` +
      `video_name=${encodeURIComponent(videoName)}&` +
      `frame_filename=${encodeURIComponent(frameFilename)}`
    );

    // Handle SSE events
    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        updateSamProgress(data);

        // If complete, process segments
        if (data.stage === 'complete') {
          holdLabelingSegments = data.segments || [];

          // Close event source
          eventSource.close();

          // Hide loading
          loadingDiv.style.display = 'none';

          // Render segments with bounding boxes and dropdowns
          renderSegmentsWithDropdowns();

          // Show submit button
          document.getElementById('btn-submit-holds').style.display = 'block';

          showStatus('step-2', `Found ${holdLabelingSegments.length} segments`, 'success');
        } else if (data.stage === 'error') {
          eventSource.close();
          throw new Error(data.message);
        }
      } catch (e) {
        console.error('Failed to parse SSE data:', e);
        eventSource.close();
        throw e;
      }
    };

    eventSource.onerror = (error) => {
      console.error('SSE error:', error);
      eventSource.close();
      throw new Error('Connection error during segmentation');
    };

  } catch (error) {
    console.error('SAM segmentation failed:', error);
    loadingDiv.style.display = 'none';
    segmentsList.innerHTML = `<p style="color: #f85149; text-align: center; margin: 0;">Error: ${error.message}</p>`;
    showStatus('step-2', `Error: ${error.message}`, 'error');
  }
}

/**
 * Update SAM segmentation progress UI
 */
function updateSamProgress(data) {
  const progressBar = document.getElementById('sam-progress-bar');
  const progressMessage = document.getElementById('sam-progress-message');

  if (progressBar && data.progress !== undefined) {
    progressBar.style.width = `${data.progress}%`;
  }

  if (progressMessage && data.message) {
    progressMessage.textContent = data.message;

    // Update color based on stage
    if (data.stage === 'error') {
      progressMessage.style.color = '#f85149';
    } else if (data.stage === 'complete') {
      progressMessage.style.color = '#3fb950';
    } else {
      progressMessage.style.color = '#aaa';
    }
  }
}

// Hold types mapping
const HOLD_TYPES = {
  '': 'Not a hold',
  'crimp': 'Crimp',
  'sloper': 'Sloper',
  'jug': 'Jug',
  'pinch': 'Pinch',
  'foot_only': 'Foot Only',
  'volume': 'Volume',
};

// Hold colors
const HOLD_COLORS = [
  { value: 'red', label: '🔴 Red' },
  { value: 'blue', label: '🔵 Blue' },
  { value: 'green', label: '🟢 Green' },
  { value: 'yellow', label: '🟡 Yellow' },
  { value: 'purple', label: '🟣 Purple' },
  { value: 'black', label: '⚫ Black' },
];

// Color palette for segments
const SEGMENT_COLORS = [
  '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
  '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B88B', '#AAB7B8',
];

/**
 * Render segments with bounding boxes on canvas and dropdowns in list
 */
function renderSegmentsWithDropdowns() {
  if (!holdLabelingImage || !holdLabelingCtx) return;

  // Clear canvas and redraw image
  holdLabelingCtx.clearRect(0, 0, holdLabelingCanvas.width, holdLabelingCanvas.height);
  holdLabelingCtx.drawImage(holdLabelingImage, 0, 0);

  // Draw bounding boxes
  holdLabelingSegments.forEach((segment, idx) => {
    const [x1, y1, x2, y2] = segment.bbox;
    const color = SEGMENT_COLORS[idx % SEGMENT_COLORS.length];

    // Draw bounding box
    holdLabelingCtx.strokeStyle = color;
    holdLabelingCtx.lineWidth = 3;
    holdLabelingCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    // Draw segment number
    holdLabelingCtx.fillStyle = color;
    holdLabelingCtx.font = 'bold 14px Arial';
    holdLabelingCtx.fillText(`#${idx + 1}`, x1 + 5, y1 + 20);
  });

  // Render segments list with dropdowns
  const segmentsList = document.getElementById('segments-list');
  if (holdLabelingSegments.length === 0) {
    segmentsList.innerHTML = '<p style="color: #888; text-align: center; margin: 0;">No segments found.</p>';
    return;
  }

  segmentsList.innerHTML = holdLabelingSegments.map((segment, idx) => {
    const color = SEGMENT_COLORS[idx % SEGMENT_COLORS.length];
    return `
      <div class="segment-item" data-segment-id="${segment.segment_id}" style="margin-bottom: 15px; padding: 12px; background: #1a1a1a; border-radius: 4px; border-left: 3px solid ${color}; display: flex; gap: 15px;">
        <!-- Left: Segment number and score (vertical) -->
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; min-width: 60px; padding-right: 15px; border-right: 1px solid #333;">
          <span style="color: #fff; font-weight: bold; font-size: 18px; margin-bottom: 5px;">#${idx + 1}</span>
          <span style="color: #aaa; font-size: 11px; text-align: center;">Score</span>
          <span style="color: #888; font-size: 12px; text-align: center;">${segment.stability_score?.toFixed(2) || 'N/A'}</span>
        </div>
        <!-- Right: Hold type and color (vertical) - takes 2/3 of remaining space -->
        <div style="flex: 0 0 66.67%; max-width: 66.67%; display: flex; flex-direction: column; gap: 10px;">
          <div>
            <label style="color: #aaa; font-size: 12px; display: block; margin-bottom: 5px;">Hold Type</label>
            <select class="hold-type-selector" data-segment-id="${segment.segment_id}" style="width: 100%; padding: 8px; background: #2a2a2a; color: #fff; border: 1px solid #444; border-radius: 4px;">
              ${Object.entries(HOLD_TYPES).map(([value, label]) =>
      `<option value="${value}">${label}</option>`
    ).join('')}
            </select>
          </div>
          <div>
            <label style="color: #aaa; font-size: 12px; display: block; margin-bottom: 5px;">Hold Color</label>
            <select class="hold-color-selector" data-segment-id="${segment.segment_id}" style="width: 100%; padding: 8px; background: #2a2a2a; color: #fff; border: 1px solid #444; border-radius: 4px;">
              ${HOLD_COLORS.map(color =>
      `<option value="${color.value}">${color.label}</option>`
    ).join('')}
            </select>
          </div>
        </div>
      </div>
    `;
  }).join('');

  // Add event listeners to dropdowns
  segmentsList.querySelectorAll('.hold-type-selector').forEach(select => {
    select.addEventListener('change', (e) => {
      const segmentId = e.target.dataset.segmentId;
      const segment = holdLabelingSegments.find(s => s.segment_id === segmentId);
      if (segment) {
        segment.hold_type = e.target.value;
        segment.is_hold = e.target.value !== '';
      }
    });
  });

  segmentsList.querySelectorAll('.hold-color-selector').forEach(select => {
    select.addEventListener('change', (e) => {
      const segmentId = e.target.dataset.segmentId;
      const segment = holdLabelingSegments.find(s => s.segment_id === segmentId);
      if (segment) {
        segment.hold_color = e.target.value;
      }
    });
  });
}

/**
 * Submit hold labels and proceed to key frame selection
 */
async function submitHoldLabels() {
  if (!currentUploadId || !currentVideoName) {
    showStatus('step-2', 'No video loaded', 'error');
    return;
  }

  try {
    showStatus('step-2', 'Saving labels...', 'info');

    // Prepare labels data
    const labels = holdLabelingSegments.map(segment => ({
      segment_id: segment.segment_id,
      hold_type: segment.hold_type || null,
      hold_color: segment.hold_color || null,
      is_hold: segment.is_hold || false,
      bbox: segment.bbox,
    }));

    const response = await fetch('/api/workflow/save-hold-labels', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        upload_id: currentUploadId,
        video_name: currentVideoName,
        labels: labels,
      }),
    });

    if (!response.ok) {
      throw new Error(`Failed to save labels: ${response.statusText}`);
    }

    const data = await response.json();
    showStatus('step-2', 'Labels saved successfully!', 'success');
    setStepCompleted('step-2');

    // Show Step 3 and scroll to it
    const step3 = document.getElementById('step-3');
    const frameSelectionUI = document.getElementById('frame-selection-ui');
    if (step3 && frameSelectionUI) {
      step3.style.display = 'block';
      setStepActive('step-3');

      // Show frame selection UI
      frameSelectionUI.style.display = 'block';

      // Load frames for key frame selection
      await loadFramesForSelection(currentUploadId, currentVideoName);

      // Scroll to step 3
      setTimeout(() => {
        scrollToStep('step-3');
      }, 300);
    }

  } catch (error) {
    console.error('Failed to submit labels:', error);
    showStatus('step-2', `Error: ${error.message}`, 'error');
  }
}

/**
 * Scroll to a specific step smoothly
 */
function scrollToStep(stepId) {
  const stepElement = document.getElementById(stepId);
  if (stepElement) {
    stepElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

// ========== Training Pool Management ==========

/**
 * Toggle between Hold Detection Labels and Key Frame Selection Labels pools
 */
function toggleTrainingPool(poolType) {
  const btnHolds = document.getElementById('btn-pool-toggle-holds');
  const btnFrames = document.getElementById('btn-pool-toggle-frames');
  const contentHolds = document.getElementById('pool-holds-content');
  const contentFrames = document.getElementById('pool-frames-content');

  if (poolType === 'holds') {
    btnHolds.style.background = '#0066cc';
    btnHolds.style.color = 'white';
    btnFrames.style.background = '#333';
    btnFrames.style.color = '#aaa';
    contentHolds.style.display = 'block';
    contentFrames.style.display = 'none';
    loadHoldDetectionPool();
  } else {
    btnHolds.style.background = '#333';
    btnHolds.style.color = '#aaa';
    btnFrames.style.background = '#0066cc';
    btnFrames.style.color = 'white';
    contentHolds.style.display = 'none';
    contentFrames.style.display = 'block';
    loadKeyFrameSelectionPool();
  }
}

/**
 * Load Hold Detection Labels Pool
 */
async function loadHoldDetectionPool() {
  try {
    const response = await fetch('/api/workflow/pool/hold-detection');
    if (!response.ok) {
      console.warn('Failed to load hold detection pool');
      return;
    }

    const data = await response.json();
    document.getElementById('pool-holds-count').textContent = data.total_sets || 0;

    const listContainer = document.getElementById('pool-holds-list');
    if (data.sets && data.sets.length > 0) {
      listContainer.innerHTML = data.sets.map(set => `
        <div style="padding: 10px; margin-bottom: 8px; background: #1a1a1a; border-radius: 4px; border-left: 3px solid #0066cc;">
          <div style="color: #fff; font-weight: bold; margin-bottom: 5px;">${set.name || set.id}</div>
          <div style="color: #aaa; font-size: 12px;">
            ${set.labeled_segments || 0} labeled segments | ${set.created_at || ''}
          </div>
        </div>
      `).join('');
    } else {
      listContainer.innerHTML = '<p style="color: #888; text-align: center; padding: 20px;">No labeled holds in pool yet.</p>';
    }
  } catch (error) {
    console.error('Failed to load hold detection pool:', error);
  }
}

/**
 * Load Key Frame Selection Labels Pool
 */
async function loadKeyFrameSelectionPool() {
  try {
    const response = await fetch('/api/workflow/pool/key-frame-selection');
    if (!response.ok) {
      console.warn('Failed to load key frame selection pool');
      return;
    }

    const data = await response.json();
    document.getElementById('pool-video-count').textContent = data.video_count || 0;
    document.getElementById('pool-frame-count').textContent = data.frame_count || 0;

    const listContainer = document.getElementById('pool-frames-list');
    if (data.videos && data.videos.length > 0) {
      listContainer.innerHTML = data.videos.map(video => `
        <div style="padding: 10px; margin-bottom: 8px; background: #1a1a1a; border-radius: 4px; border-left: 3px solid #0066cc;">
          <div style="color: #fff; font-weight: bold; margin-bottom: 5px;">${video.name || video.id}</div>
          <div style="color: #aaa; font-size: 12px;">
            ${video.selected_frames || 0} selected frames | ${video.created_at || ''}
          </div>
        </div>
      `).join('');
    } else {
      listContainer.innerHTML = '<p style="color: #888; text-align: center; padding: 20px;">No selected frames in pool yet.</p>';
    }
  } catch (error) {
    console.error('Failed to load key frame selection pool:', error);
  }
}

/**
 * Train YOLO from Hold Detection Pool
 */
async function trainYoloFromPool() {
  if (!confirm('Train YOLO model using all labeled holds in the pool?')) {
    return;
  }

  try {
    showStatus('step-2', 'Starting YOLO training from pool...', 'info');

    const response = await fetch('/api/yolo/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dataset_yaml: 'data/hold_detection_pool/dataset.yaml',
        model: 'yolov8n.pt',
        epochs: 100,
        batch: 16,
        imgsz: 640,
        upload_to_gcs: false,
      }),
    });

    if (!response.ok) {
      throw new Error(`Training failed: ${response.statusText}`);
    }

    const data = await response.json();
    showStatus('step-2', `Training started (Job ID: ${data.job_id})`, 'success');
    loadTrainingJobs();

  } catch (error) {
    console.error('Failed to start training:', error);
    showStatus('step-2', `Error: ${error.message}`, 'error');
  }
}

/**
 * Train Frame Selector from Key Frame Selection Pool
 */
async function trainFrameSelectorFromPool() {
  if (!confirm('Train frame selector model using all selected frames in the pool?')) {
    return;
  }

  try {
    showStatus('step-3', 'Starting frame selector training from pool...', 'info');

    // This would need a new endpoint that trains from the pool
    // For now, show a message
    showStatus('step-3', 'Training from pool not yet implemented. Use individual video training.', 'info');

  } catch (error) {
    console.error('Failed to start training:', error);
    showStatus('step-3', `Error: ${error.message}`, 'error');
  }
}

/**
 * Handle video file selection - extract first frame for preview
 */
function handleVideoFileSelection(event) {
  const file = event.target.files[0];
  if (!file) {
    document.getElementById('video-preview-container').style.display = 'none';
    return;
  }

  // Show loading state
  const container = document.getElementById('video-preview-container');
  container.style.display = 'block';
  document.getElementById('preview-filename').textContent = file.name;
  document.getElementById('first-frame-preview').src = '';

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
          firstFrameImageUrl = previewUrl; // Store for hold detection

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
    };

    video.load();
  } catch (error) {
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
  // Also update state variables
  holdColor = holdColorSelect.value;
  routeDifficulty = routeDifficultySelect.value;
}

/**
 * Auto-generate session name from current date/time
 */
function generateSessionName() {
  const now = new Date();
  const year = now.getFullYear();
  const month = String(now.getMonth() + 1).padStart(2, '0');
  const day = String(now.getDate()).padStart(2, '0');
  const hours = String(now.getHours()).padStart(2, '0');
  const minutes = String(now.getMinutes()).padStart(2, '0');

  return `${year}-${month}-${day}_${hours}${minutes}`;
}

/**
 * Handle convert to 3D button click
 */
async function handleConvertTo3D() {
  // Try to get upload ID from current frame directory if not set
  if (!currentUploadId && currentFrameDir) {
    // Extract upload ID from frame directory path (look for 'data' directory)
    const pathParts = currentFrameDir.split('/');
    const dataIdx = pathParts.indexOf('data');
    if (dataIdx >= 0 && dataIdx < pathParts.length - 1) {
      currentUploadId = pathParts[dataIdx + 1];
    } else {
      // Fallback: try 'storage' or 'uploads' for backward compatibility
      const storageIdx = pathParts.indexOf('storage');
      if (storageIdx >= 0 && storageIdx < pathParts.length - 1) {
        currentUploadId = pathParts[storageIdx + 1];
      } else {
        const uploadIdx = pathParts.indexOf('uploads');
        if (uploadIdx >= 0 && uploadIdx < pathParts.length - 1) {
          currentUploadId = pathParts[uploadIdx + 1];
        }
      }
    }
  }

  if (!currentUploadId) {
    show3DStatus('Please upload a video and extract frames first', 'error');
    return;
  }

  const btn = document.getElementById('btn-convert-to-3d');
  const statusDiv = document.getElementById('status-3d-conversion');

  btn.disabled = true;
  btn.textContent = '🔄 Converting...';
  statusDiv.style.display = 'block';
  show3DStatus('Starting 3D conversion...', 'info');

  try {
    const response = await fetch('/api/detection/convert-to-3d', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        upload_id: currentUploadId,
        use_sam: true,
        depth_scale: 1.0,
        mesh_resolution: null,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Conversion failed');
    }

    const data = await response.json();
    current3DModelId = data.model_id;

    show3DStatus('Conversion started. Checking status...', 'info');

    // Start polling for completion
    poll3DConversionStatus(data.model_id);

  } catch (error) {
    console.error('3D conversion failed:', error);
    show3DStatus(`Error: ${error.message}`, 'error');
    btn.disabled = false;
    btn.textContent = '🎨 Convert to 3D';
  }
}

/**
 * Poll for 3D conversion status
 */
async function poll3DConversionStatus(modelId) {
  const maxAttempts = 60; // 5 minutes max (5 second intervals)
  let attempts = 0;

  const poll = async () => {
    attempts++;

    try {
      const response = await fetch(`/api/detection/3d-models/${modelId}`);

      if (!response.ok) {
        if (response.status === 404 && attempts < maxAttempts) {
          // Still processing, check again
          setTimeout(poll, 5000);
          show3DStatus(`Processing... (${attempts * 5}s)`, 'info');
          return;
        }
        throw new Error('Failed to check conversion status');
      }

      const metadata = await response.json();

      // Check if there's an error
      if (metadata.status === 'error') {
        show3DStatus(`Error: ${metadata.error}`, 'error');
        resetConvertButton();
        return;
      }

      // Conversion complete
      show3DStatus('Conversion complete! Loading 3D model...', 'success');
      await display3DModel(modelId);
      resetConvertButton();

    } catch (error) {
      if (attempts < maxAttempts) {
        setTimeout(poll, 5000);
      } else {
        console.error('Polling failed:', error);
        show3DStatus('Conversion timeout. Please check server logs.', 'error');
        resetConvertButton();
      }
    }
  };

  poll();
}

/**
 * Display 3D model in viewer
 */
async function display3DModel(modelId) {
  try {
    // Get preview data
    const response = await fetch(`/api/detection/3d-models/${modelId}/preview`);
    if (!response.ok) {
      throw new Error('Failed to get 3D model preview');
    }

    const data = await response.json();

    if (!data.mesh_url) {
      throw new Error('No mesh file available');
    }

    // Show viewer container
    const viewerContainer = document.getElementById('3d-viewer-container');
    viewerContainer.style.display = 'block';

    // Initialize Three.js viewer if not already done
    if (!threeViewer) {
      const viewerDiv = document.getElementById('three-viewer');
      threeViewer = new ThreeViewer('three-viewer', {
        width: viewerDiv.clientWidth,
        height: 500,
        backgroundColor: 0x1a1a1a,
      });
    }

    // Load model
    await threeViewer.loadModel(data.mesh_url, data.mesh_file);

    // Update model info
    const infoDiv = document.getElementById('3d-model-info');
    if (data.metadata && data.metadata.mesh_stats) {
      const stats = data.metadata.mesh_stats;
      infoDiv.textContent = `Vertices: ${stats.vertex_count.toLocaleString()}, Faces: ${stats.face_count.toLocaleString()}`;
    }

    show3DStatus('3D model loaded successfully!', 'success');

    // Scroll to viewer
    viewerContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  } catch (error) {
    console.error('Failed to display 3D model:', error);
    show3DStatus(`Error loading 3D model: ${error.message}`, 'error');
  }
}

/**
 * Close 3D viewer
 */
function close3DViewer() {
  const viewerContainer = document.getElementById('3d-viewer-container');
  viewerContainer.style.display = 'none';

  if (threeViewer) {
    threeViewer.destroy();
    threeViewer = null;
  }

  current3DModelId = null;
}

/**
 * Show 3D conversion status
 */
function show3DStatus(message, type) {
  const statusDiv = document.getElementById('status-3d-conversion');
  statusDiv.textContent = message;
  statusDiv.className = `status-3d-${type}`;

  // Set color based on type
  if (type === 'error') {
    statusDiv.style.background = 'rgba(218, 54, 51, 0.2)';
    statusDiv.style.color = '#f85149';
    statusDiv.style.border = '1px solid #f85149';
  } else if (type === 'success') {
    statusDiv.style.background = 'rgba(35, 134, 54, 0.2)';
    statusDiv.style.color = '#3fb950';
    statusDiv.style.border = '1px solid #3fb950';
  } else {
    statusDiv.style.background = 'rgba(56, 139, 253, 0.2)';
    statusDiv.style.color = '#58a6ff';
    statusDiv.style.border = '1px solid #58a6ff';
  }
}

/**
 * Reset convert button
 */
function resetConvertButton() {
  const btn = document.getElementById('btn-convert-to-3d');
  btn.disabled = false;
  btn.textContent = '🎨 Convert to 3D';
}

