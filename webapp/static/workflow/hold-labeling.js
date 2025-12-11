/**
 * Hold Labeling Module
 * 
 * Handles SAM-based hold labeling functionality.
 */

// Hold types mapping
const HOLD_TYPES = {
  '': 'Not selected',
  'crimp': 'Crimp',
  'sloper': 'Sloper',
  'jug': 'Jug',
  'pinch': 'Pinch',
  'foot_only': 'Foot Only',
  'volume': 'Volume',
};

// Hold colors
const HOLD_COLORS = [
  { value: '', label: 'Not selected' },
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
 * Show hold labeling UI after frame extraction in hold detection mode
 * @param {string} uploadId - Upload ID
 * @param {string} videoName - Video name
 * @param {number} frameCount - Frame count
 */
function showHoldLabelingUI(uploadId, videoName, frameCount) {
  // Get hold color and difficulty from selectors
  const holdColorSelect = document.getElementById('hold-color');
  const routeDifficultySelect = document.getElementById('route-difficulty');

  if (holdColorSelect) {
    WorkflowState.setHoldColor(holdColorSelect.value);
  }
  if (routeDifficultySelect) {
    WorkflowState.setRouteDifficulty(routeDifficultySelect.value);
  }

  // Show the hold labeling UI
  const holdLabelingUI = document.getElementById('hold-labeling-ui');
  if (holdLabelingUI) {
    holdLabelingUI.style.display = 'block';
  }

  // Initialize canvas
  const canvas = document.getElementById('hold-labeling-canvas');
  if (canvas) {
    WorkflowState.holdLabelingCanvas = canvas;
    WorkflowState.holdLabelingCtx = canvas.getContext('2d');

    // Add click event listener for bounding box selection
    canvas.addEventListener('click', handleCanvasClick);
  }

  // Load first frame for labeling and automatically start SAM segmentation
  loadFirstFrameForLabeling(uploadId, videoName);
}

/**
 * Load first frame for hold labeling and start SAM segmentation automatically
 * @param {string} uploadId - Upload ID
 * @param {string} videoName - Video name
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
 * @param {string} imageUrl - Image URL
 * @returns {Promise} Promise that resolves when image is loaded
 */
function loadImageToCanvas(imageUrl) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      WorkflowState.holdLabelingImage = img;
      const canvas = WorkflowState.holdLabelingCanvas;
      const ctx = WorkflowState.holdLabelingCtx;

      if (!canvas || !ctx) {
        reject(new Error('Canvas not initialized'));
        return;
      }

      // Get container dimensions (the inner div that holds the canvas)
      const container = canvas.parentElement; // Get the inner container div
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
      canvas.style.width = `${displayWidth}px`;
      canvas.style.height = `${displayHeight}px`;
      canvas.style.maxWidth = '100%';
      canvas.style.maxHeight = `${maxHeight}px`;

      // Set canvas internal size to match image (for drawing)
      canvas.width = img.width;
      canvas.height = img.height;

      // Draw image
      ctx.drawImage(img, 0, 0);
      resolve();
    };
    img.onerror = reject;
    img.src = imageUrl;
  });
}

/**
 * Start SAM segmentation on first frame with real-time progress
 * @param {string} uploadId - Upload ID
 * @param {string} videoName - Video name
 * @param {string} frameFilename - Frame filename
 */
async function startSamSegmentation(uploadId, videoName, frameFilename) {
  const loadingDiv = document.getElementById('sam-loading');
  const segmentsList = document.getElementById('segments-list');

  if (!loadingDiv || !segmentsList) {
    console.error('SAM UI elements not found');
    return;
  }

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
          WorkflowState.holdLabelingSegments = data.segments || [];

          // Close event source
          eventSource.close();

          // Hide loading
          loadingDiv.style.display = 'none';

          // Render segments with bounding boxes and dropdowns
          renderSegmentsWithDropdowns();

          // Show submit button
          const btnSubmitHolds = document.getElementById('btn-submit-holds');
          if (btnSubmitHolds) {
            btnSubmitHolds.style.display = 'block';
          }

          showStatus('step-2', `Found ${WorkflowState.holdLabelingSegments.length} segments`, 'success');
          if (typeof updateDashboardStatus === 'function') {
            updateDashboardStatus();
          }
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
    if (loadingDiv) {
      loadingDiv.style.display = 'none';
    }
    if (segmentsList) {
      segmentsList.innerHTML = `<p style="color: #f85149; text-align: center; margin: 0;">Error: ${error.message}</p>`;
    }
    showStatus('step-2', `Error: ${error.message}`, 'error');
  }
}

/**
 * Update SAM segmentation progress UI
 * @param {Object} data - Progress data
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

/**
 * Handle canvas click for segment selection
 * @param {MouseEvent} event - Click event
 */
function handleCanvasClick(event) {
  const canvas = WorkflowState.holdLabelingCanvas;
  const segments = WorkflowState.holdLabelingSegments;

  if (!canvas || !segments.length) return;

  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;

  // Scale to canvas coordinates (account for display size vs internal size)
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const canvasX = x * scaleX;
  const canvasY = y * scaleY;

  // Find clicked segment (check from last to first for proper z-order)
  let clickedSegment = null;
  for (let i = segments.length - 1; i >= 0; i--) {
    const segment = segments[i];
    const [x1, y1, x2, y2] = segment.bbox;
    if (canvasX >= x1 && canvasX <= x2 && canvasY >= y1 && canvasY <= y2) {
      clickedSegment = segment;
      break;
    }
  }

  // Select the clicked segment
  if (clickedSegment) {
    WorkflowState.selectedSegmentId = clickedSegment.segment_id;
    // Highlight the selected segment in the dropdown list
    highlightSelectedSegment(clickedSegment.segment_id);
    // Re-render to show selection
    renderSegmentsWithDropdowns();
  } else {
    // Deselect if clicked outside
    WorkflowState.selectedSegmentId = null;
    highlightSelectedSegment(null);
    renderSegmentsWithDropdowns();
  }
}

/**
 * Highlight selected segment in the dropdown list
 * @param {string|null} segmentId - Segment ID or null
 */
function highlightSelectedSegment(segmentId) {
  // Remove previous highlights
  document.querySelectorAll('.segment-item').forEach(item => {
    // 원래 스타일로 복원
    item.style.border = item.style.borderLeft;
    item.style.backgroundColor = '';
    item.style.boxShadow = '';
  });

  // Highlight selected segment
  if (segmentId) {
    const selectedItem = document.querySelector(`[data-segment-id="${segmentId}"]`);
    if (selectedItem) {
      // 하이라이트 스타일 적용
      selectedItem.style.border = '4px solid #0066cc';
      selectedItem.style.borderLeft = selectedItem.style.border;
      selectedItem.style.backgroundColor = 'rgba(0, 102, 204, 0.15)';
      // Glow effect를 더 강하게 - 여러 레이어의 그림자 추가
      selectedItem.style.boxShadow = '0 0 0 2px rgba(0, 102, 204, 0.3), 0 0 20px rgba(0, 102, 204, 0.5), 0 0 40px rgba(0, 102, 204, 0.3)';

      // Scroll into view - block: 'center'로 변경하여 카드가 중앙에 오도록
      selectedItem.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }
}

/**
 * Render segments with bounding boxes on canvas and dropdowns in list
 */
function renderSegmentsWithDropdowns() {
  const image = WorkflowState.holdLabelingImage;
  const canvas = WorkflowState.holdLabelingCanvas;
  const ctx = WorkflowState.holdLabelingCtx;
  const segments = WorkflowState.holdLabelingSegments;

  if (!image || !ctx || !canvas) return;

  // Clear canvas and redraw image
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(image, 0, 0);

  // Draw bounding boxes
  segments.forEach((segment, idx) => {
    const [x1, y1, x2, y2] = segment.bbox;
    const color = SEGMENT_COLORS[idx % SEGMENT_COLORS.length];
    const isSelected = segment.segment_id === WorkflowState.selectedSegmentId;

    // Draw bounding box with thicker border if selected
    ctx.strokeStyle = isSelected ? '#0066cc' : color;
    ctx.lineWidth = isSelected ? 5 : 3;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    // Draw segment number
    ctx.fillStyle = isSelected ? '#0066cc' : color;
    ctx.font = 'bold 14px Arial';
    ctx.fillText(`#${idx + 1}`, x1 + 5, y1 + 20);
  });

  // Render segments list with dropdowns
  const segmentsList = document.getElementById('segments-list');
  if (!segmentsList) return;

  if (segments.length === 0) {
    segmentsList.innerHTML = '<p style="color: #888; text-align: center; margin: 0;">No segments found.</p>';
    return;
  }

  segmentsList.innerHTML = segments.map((segment, idx) => {
    const color = SEGMENT_COLORS[idx % SEGMENT_COLORS.length];
    return `
      <div class="segment-item" data-segment-id="${segment.segment_id}" style="margin-bottom: 15px; padding: 12px; background: #1a1a1a; border-radius: 4px; border-left: 3px solid ${color}; display: flex; gap: 15px;">
        <!-- Left: Segment number and score (vertical) -->
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; min-width: 80px; padding-right: 15px; border-right: 1px solid #333;">
          <span style="color: #fff; font-weight: bold; font-size: 18px; margin-bottom: 5px;">#${idx + 1}</span>
          <span style="color: #aaa; font-size: 11px; text-align: center; white-space: nowrap;">Score: ${segment.stability_score?.toFixed(2) || 'N/A'}</span>
        </div>
        <!-- Right: Hold type and color (vertical) - takes remaining space -->
        <div style="flex: 1; display: flex; flex-direction: column; gap: 10px;">
          <div>
            <label style="color: #aaa; font-size: 12px; display: block; margin-bottom: 5px;">Hold Type</label>
            <select class="hold-type-selector" data-segment-id="${segment.segment_id}" style="width: 100%; padding: 8px; background: #2a2a2a; color: #fff; border: 1px solid #444; border-radius: 4px;">
              ${Object.entries(HOLD_TYPES).map(([value, label]) =>
      `<option value="${value}" ${value === '' ? 'selected' : ''}>${label}</option>`
    ).join('')}
            </select>
          </div>
          <div>
            <label style="color: #aaa; font-size: 12px; display: block; margin-bottom: 5px;">Hold Color</label>
            <select class="hold-color-selector" data-segment-id="${segment.segment_id}" style="width: 100%; padding: 8px; background: #2a2a2a; color: #fff; border: 1px solid #444; border-radius: 4px;">
              ${HOLD_COLORS.map(color =>
      `<option value="${color.value}" ${color.value === '' ? 'selected' : ''}>${color.label}</option>`
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
      const segment = segments.find(s => s.segment_id === segmentId);
      if (segment) {
        segment.hold_type = e.target.value;
        segment.is_hold = e.target.value !== '';
      }
    });
  });

  segmentsList.querySelectorAll('.hold-color-selector').forEach(select => {
    select.addEventListener('change', (e) => {
      const segmentId = e.target.dataset.segmentId;
      const segment = segments.find(s => s.segment_id === segmentId);
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
  if (!WorkflowState.getCurrentUploadId() || !WorkflowState.getCurrentVideoName()) {
    showStatus('step-2', 'No video loaded', 'error');
    return;
  }

  try {
    showStatus('step-2', 'Saving labels...', 'info');

    // Prepare labels data
    const segments = WorkflowState.holdLabelingSegments;
    const labels = segments.map(segment => ({
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
        upload_id: WorkflowState.getCurrentUploadId(),
        video_name: WorkflowState.getCurrentVideoName(),
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
      if (typeof loadFramesForSelection === 'function') {
        await loadFramesForSelection(WorkflowState.getCurrentUploadId(), WorkflowState.getCurrentVideoName());
      }

      // Automatically navigate to step 3
      setTimeout(() => {
        if (typeof navigateToStep === 'function') {
          navigateToStep('step-3');
        }
      }, 300);
    }

  } catch (error) {
    console.error('Failed to submit labels:', error);
    showStatus('step-2', `Error: ${error.message}`, 'error');
  }
}
