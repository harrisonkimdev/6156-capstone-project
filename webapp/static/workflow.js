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

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  setupEventListeners();
  loadSessions();
  loadTrainingJobs();

  // Poll for updates every 5 seconds
  setInterval(() => {
    loadSessions();
    loadTrainingJobs();
  }, 5000);
});

/**
 * Setup event listeners
 */
function setupEventListeners() {
  document.getElementById('btn-extract-frames').addEventListener('click', extractFrames);
  document.getElementById('btn-use-test-frames').addEventListener('click', useTestFrames);
  document.getElementById('btn-create-session').addEventListener('click', createSession);
  document.getElementById('btn-start-training').addEventListener('click', startTraining);
  document.getElementById('btn-upload-gcs').addEventListener('click', uploadToGCS);
  document.getElementById('btn-clear-data')?.addEventListener('click', clearAllData);

  // Pipeline mode toggle
  document.querySelectorAll('input[name="pipeline-mode"]').forEach(radio => {
    radio.addEventListener('change', handlePipelineModeChange);
  });

  // Frame selection UI
  document.getElementById('frame-slider')?.addEventListener('input', handleFrameSliderChange);
  document.getElementById('btn-train-frame-selector')?.addEventListener('click', trainFrameSelector);

  // Keyboard shortcuts for frame selection
  document.addEventListener('keydown', handleKeyboardShortcuts);
}

/**
 * Step 1: Extract frames from video
 */
async function extractFrames() {
  const videoFile = document.getElementById('video-file').files[0];
  if (!videoFile) {
    showStatus('step-1', 'Please select a video file', 'error');
    return;
  }

  const method = document.getElementById('sampling-method').value;

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

    // Update UI
    document.getElementById('frame-dir').value = currentFrameDir;
    document.getElementById('btn-create-session').disabled = false;

    showStatus('step-1', `Extracted ${data.frame_count} frames`, 'success');
    setStepCompleted('step-1');
    setStepActive('step-2');

    // Load frames for manual selection if in frame selection mode
    const pipelineMode = document.querySelector('input[name="pipeline-mode"]:checked').value;
    if (pipelineMode === 'frame_selection') {
      // Extract upload_id and video_name from frame_directory
      const pathParts = data.frame_directory.split('/');
      const uploadId = pathParts[pathParts.length - 2];
      const videoName = pathParts[pathParts.length - 1];
      await loadFramesForSelection(uploadId, videoName);
    }

  } catch (error) {
    console.error('Frame extraction failed:', error);
    showStatus('step-1', `Error: ${error.message}`, 'error');
  }
}

/**
 * Step 1B: Use pre-extracted test frames (Dev Mode)
 */
async function useTestFrames() {
  showStatus('step-1', '🧪 Loading test frames...', 'info');
  setStepActive('step-1');

  try {
    const response = await fetch('/api/workflow/use-test-frames');

    if (!response.ok) {
      throw new Error(`Failed to load test frames: ${response.statusText}`);
    }

    const data = await response.json();
    currentFrameDir = data.frame_directory;

    // Update UI
    document.getElementById('frame-dir').value = currentFrameDir;
    document.getElementById('btn-create-session').disabled = false;

    showStatus('step-1', `✅ ${data.message} (${data.frame_count} frames)`, 'success');
    setStepCompleted('step-1');
    setStepActive('step-2');

  } catch (error) {
    console.error('Test frames loading failed:', error);
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
      container.innerHTML = '<p style="color: #888;">No sessions yet. Create one in Step 2.</p>';
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
      container.innerHTML = '<p style="color: #888;">No training jobs yet. Start one in Step 3.</p>';
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
};

/**
 * Handle pipeline mode change
 */
function handlePipelineModeChange(event) {
  const mode = event.target.value;
  const frameSelectionUI = document.getElementById('frame-selection-ui');
  const step2 = document.getElementById('step-2');
  const step3 = document.getElementById('step-3');
  const step4 = document.getElementById('step-4');

  console.log(`[Pipeline Mode] Changed to: ${mode}`);

  if (!frameSelectionUI || !step2 || !step3 || !step4) {
    console.error('[Pipeline Mode] Missing UI elements');
    return;
  }

  if (mode === 'frame_selection') {
    // Show frame selection UI, hide hold detection steps
    frameSelectionUI.style.display = 'block';
    step2.style.display = 'none';
    step3.style.display = 'none';
    step4.style.display = 'none';
    console.log('[Pipeline Mode] Showing frame selection UI');
  } else {
    // Hide frame selection UI, show hold detection steps
    frameSelectionUI.style.display = 'none';
    step2.style.display = 'block';
    step3.style.display = 'block';
    step4.style.display = 'block';
    console.log('[Pipeline Mode] Showing hold detection steps');
  }
}

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
    document.getElementById('frame-selected-count').textContent = frameSelectionState.selectedFrames.size;
    document.getElementById('frame-slider').max = data.frames.length - 1;

    // Load first frame
    frameAspectRatio = null; // Reset aspect ratio detection
    updateFramePreview();

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

  // Update selected badge
  const badge = document.getElementById('frame-selected-badge');
  if (frameSelectionState.selectedFrames.has(frame.filename)) {
    badge.style.display = 'block';
  } else {
    badge.style.display = 'none';
  }

  // Update previously selected frames display
  updatePreviouslySelectedFrames();

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
  const currentIndex = frameSelectionState.currentIndex;

  // Get the most recent previously selected frame
  const previousSelectedFrames = frameSelectionState.frames
    .slice(0, currentIndex)
    .filter((f) => frameSelectionState.selectedFrames.has(f.filename));

  const lastSelectedFrame = previousSelectedFrames.length > 0
    ? previousSelectedFrames[previousSelectedFrames.length - 1]
    : null;

  if (frameAspectRatio === 'vertical') {
    const prevImg = document.getElementById('frame-preview-prev-vertical');
    if (prevImg && lastSelectedFrame) {
      prevImg.src = lastSelectedFrame.path;
    } else if (prevImg) {
      prevImg.src = '';
    }
  } else {
    const prevImg = document.getElementById('frame-preview-prev-horizontal');
    if (prevImg && lastSelectedFrame) {
      prevImg.src = lastSelectedFrame.path;
    } else if (prevImg) {
      prevImg.src = '';
    }
  }
}

/**
 * Handle frame slider change
 */
function handleFrameSliderChange(event) {
  frameSelectionState.currentIndex = parseInt(event.target.value);
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
  const newIndex = frameSelectionState.currentIndex + direction;
  if (newIndex >= 0 && newIndex < frameSelectionState.frames.length) {
    frameSelectionState.currentIndex = newIndex;
    updateFramePreview();
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
    document.getElementById('frame-selected-count').textContent = frameSelectionState.selectedFrames.size;
    
    // Move to next frame automatically after selection
    if (frameSelectionState.currentIndex < frameSelectionState.frames.length - 1) {
      frameSelectionState.currentIndex++;
    }
    
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
    document.getElementById('frame-selected-count').textContent = frameSelectionState.selectedFrames.size;
    updateFramePreview();

  } catch (error) {
    console.error('Failed to deselect frame:', error);
  }
}

/**
 * Train frame selector model
 */
async function trainFrameSelector() {
  if (frameSelectionState.selectedFrames.size === 0) {
    alert('Please select at least one frame before training');
    return;
  }

  if (!confirm(`Train frame selector model with ${frameSelectionState.selectedFrames.size} selected frames?`)) {
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
    showStatus('step-1', `Training complete! Accuracy: ${(data.accuracy * 100).toFixed(1)}%`, 'success');

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
    };

    // Reload page
    location.reload();

  } catch (error) {
    console.error('Failed to clear data:', error);
    alert(`❌ Error: ${error.message}`);
  }
}
