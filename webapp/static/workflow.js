/**
 * Hold Labeling Workflow - Frontend JavaScript
 * 
 * Manages the end-to-end workflow from video upload to model deployment.
 */

// State
let currentFrameDir = null;
let currentSessionId = null;
let currentTrainingJobId = null;

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
