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
  { value: 'white', label: '⚪ White' },
];

// Color palette for segments
const SEGMENT_COLORS = [
  '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
  '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B88B', '#AAB7B8',
];

/**
 * Show frame selection modal for choosing between original and trimmed first frame
 * @param {string} uploadId - Upload ID
 * @param {string} videoName - Video name  
 * @param {object} trimInfo - Trim info with originalFirstFramePath and trimmedFirstFramePath
 */
function showFrameSelectionModal(uploadId, videoName, trimInfo) {
  const modal = document.getElementById('frame-selection-modal');
  const originalPreview = document.getElementById('original-frame-preview');
  const trimmedPreview = document.getElementById('trimmed-frame-preview');
  const btnSelectOriginal = document.getElementById('btn-select-original-frame');
  const btnSelectTrimmed = document.getElementById('btn-select-trimmed-frame');
  const originalOption = document.getElementById('original-frame-option');
  const trimmedOption = document.getElementById('trimmed-frame-option');

  if (!modal) {
    console.error('Frame selection modal not found');
    return;
  }

  // Show modal
  modal.style.display = 'block';

  // Load preview images
  if (originalPreview && trimInfo.originalFirstFramePath) {
    originalPreview.src = trimInfo.originalFirstFramePath;
  }
  if (trimmedPreview && trimInfo.trimmedFirstFramePath) {
    trimmedPreview.src = trimInfo.trimmedFirstFramePath;
  }

  // Hover effects for frame options
  function addHoverEffect(element) {
    if (!element) return;
    element.addEventListener('mouseenter', () => {
      element.style.borderColor = '#0066cc';
    });
    element.addEventListener('mouseleave', () => {
      element.style.borderColor = 'transparent';
    });
  }
  addHoverEffect(originalOption);
  addHoverEffect(trimmedOption);

  // Handle original frame selection
  function selectOriginalFrame() {
    WorkflowState.setSelectedFrameForSegmentation('original');
    modal.style.display = 'none';

    // Load frame and start SAM segmentation
    loadFirstFrameForLabelingWithPath(
      uploadId,
      videoName,
      trimInfo.originalFirstFramePath,
      'original_first_frame.jpg'
    );

    // Remove event listeners
    if (btnSelectOriginal) btnSelectOriginal.removeEventListener('click', selectOriginalFrame);
    if (btnSelectTrimmed) btnSelectTrimmed.removeEventListener('click', selectTrimmedFrame);
  }

  // Handle trimmed frame selection
  function selectTrimmedFrame() {
    WorkflowState.setSelectedFrameForSegmentation('trimmed');
    modal.style.display = 'none';

    // Load frame and start SAM segmentation
    loadFirstFrameForLabelingWithPath(
      uploadId,
      videoName,
      trimInfo.trimmedFirstFramePath,
      'trimmed_first_frame.jpg'
    );

    // Remove event listeners
    if (btnSelectOriginal) btnSelectOriginal.removeEventListener('click', selectOriginalFrame);
    if (btnSelectTrimmed) btnSelectTrimmed.removeEventListener('click', selectTrimmedFrame);
  }

  // Add click listeners
  if (btnSelectOriginal) {
    btnSelectOriginal.addEventListener('click', selectOriginalFrame);
  }
  if (btnSelectTrimmed) {
    btnSelectTrimmed.addEventListener('click', selectTrimmedFrame);
  }
}

/**
 * Load first frame for labeling with a specific path and start SAM segmentation
 * @param {string} uploadId - Upload ID
 * @param {string} videoName - Video name
 * @param {string} framePath - Path to the frame image
 * @param {string} frameFilename - Filename for SAM segmentation
 */
async function loadFirstFrameForLabelingWithPath(uploadId, videoName, framePath, frameFilename) {
  try {
    // Show hold labeling UI
    const holdLabelingUI = document.getElementById('hold-labeling-ui');
    if (holdLabelingUI) {
      holdLabelingUI.style.display = 'block';
    }

    // Initialize canvas if not already done
    const canvas = document.getElementById('hold-labeling-canvas');
    if (canvas && !WorkflowState.holdLabelingCanvas) {
      WorkflowState.holdLabelingCanvas = canvas;
      WorkflowState.holdLabelingCtx = canvas.getContext('2d');
      canvas.addEventListener('click', handleCanvasClick);
      canvas.setAttribute('tabindex', '0');
      canvas.addEventListener('keydown', handleCanvasKeyDown);
    }

    // Load image onto canvas
    await loadImageToCanvas(framePath);

    // Start SAM segmentation
    await startSamSegmentation(uploadId, videoName, frameFilename);
  } catch (error) {
    console.error('Failed to load frame for labeling:', error);
    showStatus('step-2', `Error loading frame: ${error.message}`, 'error');
  }
}

/**
 * Show hold labeling UI after frame extraction in hold detection mode
 * @param {string} uploadId - Upload ID
 * @param {string} videoName - Video name
 * @param {number} frameCount - Frame count
 * @param {object} trimInfo - Optional trim info with frame paths
 */
function showHoldLabelingUI(uploadId, videoName, frameCount, trimInfo = null) {
  // Get hold color and difficulty from selectors
  const holdColorSelect = document.getElementById('hold-color');
  const routeDifficultySelect = document.getElementById('route-difficulty');

  if (holdColorSelect) {
    WorkflowState.setHoldColor(holdColorSelect.value);
  }
  if (routeDifficultySelect) {
    WorkflowState.setRouteDifficulty(routeDifficultySelect.value);
  }

  // Check if video was trimmed and both frame paths are available
  if (trimInfo && trimInfo.hasTrimmed &&
    trimInfo.originalFirstFramePath && trimInfo.trimmedFirstFramePath) {
    // Show frame selection modal - let user choose which frame to use
    showFrameSelectionModal(uploadId, videoName, trimInfo);
  } else {
    // No trim or no trimmed frame available - show hold labeling UI directly
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

      // Add keyboard event listener for Backspace to remove selected segments
      canvas.setAttribute('tabindex', '0'); // Make canvas focusable
      canvas.addEventListener('keydown', handleCanvasKeyDown);
    }

    // Load first frame for labeling and automatically start SAM segmentation
    loadFirstFrameForLabeling(uploadId, videoName);
  }
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

  // Update progress bar
  if (progressBar) {
    const progress = data.progress !== undefined ? data.progress : 0;
    progressBar.style.width = `${progress}%`;
  }

  // Update progress message
  if (progressMessage) {
    if (data.message) {
      progressMessage.textContent = data.message;
    }

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

  // Focus canvas to receive keyboard events
  canvas.focus();

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

  // Select the clicked segment (support multi-select with Shift)
  if (clickedSegment) {
    const selectedSet = WorkflowState.selectedSegmentIds || new Set();
    if (event.shiftKey) {
      if (selectedSet.has(clickedSegment.segment_id)) {
        selectedSet.delete(clickedSegment.segment_id);
      } else {
        selectedSet.add(clickedSegment.segment_id);
      }
      WorkflowState.selectedSegmentIds = selectedSet;
      WorkflowState.selectedSegmentId = clickedSegment.segment_id; // keep single for compatibility
    } else {
      selectedSet.clear();
      selectedSet.add(clickedSegment.segment_id);
      WorkflowState.selectedSegmentIds = selectedSet;
      WorkflowState.selectedSegmentId = clickedSegment.segment_id;
    }
    highlightSelectedSegments(clickedSegment.segment_id);
    renderSegmentsWithDropdowns();
  } else {
    // Deselect if clicked outside (unless holding shift)
    if (!event.shiftKey && WorkflowState.selectedSegmentIds) {
      WorkflowState.selectedSegmentIds.clear();
      WorkflowState.selectedSegmentId = null;
    }
    highlightSelectedSegments(null);
    renderSegmentsWithDropdowns();
  }
}

/**
 * Handle keyboard events on canvas (Backspace to mark as "not a hold")
 * @param {KeyboardEvent} event - Keyboard event
 */
function handleCanvasKeyDown(event) {
  // Backspace or Delete key: mark selected segments as "not a hold"
  if (event.key === 'Backspace' || event.key === 'Delete') {
    event.preventDefault();

    const selectedSet = WorkflowState.selectedSegmentIds || new Set();
    const segments = WorkflowState.holdLabelingSegments;

    // Get segments to mark as "not a hold"
    const idsToMark = selectedSet.size > 0
      ? Array.from(selectedSet)
      : (WorkflowState.selectedSegmentId ? [WorkflowState.selectedSegmentId] : []);

    // Mark segments as "not a hold" (same as selecting "Not selected" in dropdown)
    idsToMark.forEach(segmentId => {
      const segment = segments.find(s => s.segment_id === segmentId);
      if (segment) {
        segment.hold_type = '';
        segment.is_hold = false;
      }
    });

    renderSegmentsWithDropdowns();
  }
}

/**
 * Highlight selected segments in the dropdown list (supports multi-select)
 * @param {string|null} scrollToId - Segment ID to scroll into view (last clicked), or null
 */
function highlightSelectedSegments(scrollToId = null) {
  const selectedSet = WorkflowState.selectedSegmentIds || new Set();

  // Include single selected segment ID if no multi-select
  if (selectedSet.size === 0 && WorkflowState.selectedSegmentId) {
    selectedSet.add(WorkflowState.selectedSegmentId);
  }

  // Remove previous highlights
  document.querySelectorAll('.segment-item').forEach(item => {
    // Restore original border-left style (from inline style in renderSegmentsWithDropdowns)
    const segmentId = item.getAttribute('data-segment-id');
    const segments = WorkflowState.holdLabelingSegments || [];
    const segment = segments.find(s => s.segment_id === segmentId);
    if (segment) {
      const segmentIndex = segments.indexOf(segment);
      const color = SEGMENT_COLORS[segmentIndex % SEGMENT_COLORS.length];
      item.style.border = 'none';
      item.style.borderLeft = `3px solid ${color}`;
    }
    item.style.backgroundColor = '';
    item.style.boxShadow = '';
  });

  // Highlight selected segments with enhanced visual effect
  selectedSet.forEach(id => {
    const selectedItem = document.querySelector(`[data-segment-id="${id}"]`);
    if (selectedItem) {
      selectedItem.style.border = '4px solid #0066cc';
      selectedItem.style.borderLeft = '4px solid #0066cc';
      selectedItem.style.backgroundColor = 'rgba(0, 102, 204, 0.3)'; // More visible background
      selectedItem.style.boxShadow = '0 0 0 2px rgba(0, 102, 204, 0.5), 0 0 20px rgba(0, 102, 204, 0.6), 0 0 40px rgba(0, 102, 204, 0.4)';
      selectedItem.style.transition = 'all 0.3s ease';

      // Add pulsing animation effect
      selectedItem.style.animation = 'pulse-highlight 2s ease-in-out infinite';

      // Scroll into view for the last clicked
      if (scrollToId && scrollToId === id) {
        selectedItem.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }
  });

  // Add CSS animation if not already added
  if (!document.getElementById('segment-highlight-style')) {
    const style = document.createElement('style');
    style.id = 'segment-highlight-style';
    style.textContent = `
      @keyframes pulse-highlight {
        0%, 100% {
          box-shadow: 0 0 0 2px rgba(0, 102, 204, 0.5), 0 0 20px rgba(0, 102, 204, 0.6), 0 0 40px rgba(0, 102, 204, 0.4);
        }
        50% {
          box-shadow: 0 0 0 3px rgba(0, 102, 204, 0.7), 0 0 30px rgba(0, 102, 204, 0.8), 0 0 60px rgba(0, 102, 204, 0.6);
        }
      }
    `;
    document.head.appendChild(style);
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

  // Draw bounding boxes (only for segments that are holds)
  segments.forEach((segment, idx) => {
    // Skip segments explicitly marked as "not a hold"
    // Only hide if is_hold is explicitly false (user marked it as not a hold)
    if (segment.is_hold === false) {
      return;
    }

    const [x1, y1, x2, y2] = segment.bbox;
    const color = SEGMENT_COLORS[idx % SEGMENT_COLORS.length];
    const isSelected =
      (WorkflowState.selectedSegmentIds && WorkflowState.selectedSegmentIds.has(segment.segment_id)) ||
      segment.segment_id === WorkflowState.selectedSegmentId;

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
      `<option value="${value}" ${(segment.hold_type || '') === value ? 'selected' : ''}>${label}</option>`
    ).join('')}
            </select>
          </div>
          <div>
            <label style="color: #aaa; font-size: 12px; display: block; margin-bottom: 5px;">Hold Color</label>
            <select class="hold-color-selector" data-segment-id="${segment.segment_id}" style="width: 100%; padding: 8px; background: #2a2a2a; color: #fff; border: 1px solid #444; border-radius: 4px;">
              ${HOLD_COLORS.map(color =>
      `<option value="${color.value}" ${(segment.hold_color || '') === color.value ? 'selected' : ''}>${color.label}</option>`
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

  // Highlight selected segments in the list
  highlightSelectedSegments();
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

    // Get hold color and route difficulty from WorkflowState
    const holdColor = WorkflowState.holdColor || '';
    const routeDifficulty = WorkflowState.routeDifficulty || '';

    const response = await fetch('/api/workflow/save-hold-labels', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        upload_id: WorkflowState.getCurrentUploadId(),
        video_name: WorkflowState.getCurrentVideoName(),
        labels: labels,
        hold_color: holdColor || undefined,
        route_difficulty: routeDifficulty || undefined,
      }),
    });

    if (!response.ok) {
      throw new Error(`Failed to save labels: ${response.statusText}`);
    }

    const data = await response.json();
    showStatus('step-2', 'Labels saved successfully!', 'success');

    // Mark labels as submitted
    if (WorkflowState.setHoldLabelsSubmitted) {
      WorkflowState.setHoldLabelsSubmitted(true);
    }

    setStepCompleted('step-2');
    if (typeof updateDashboardStatus === 'function') {
      updateDashboardStatus();
    }

    // Show Step 3 and scroll to it
    const step3 = document.getElementById('step-3');
    const frameSelectionUI = document.getElementById('frame-selection-ui');
    if (step3 && frameSelectionUI) {
      step3.style.display = 'block';
      setStepActive('step-3');

      // Show frame selection UI
      frameSelectionUI.style.display = 'block';

      // Show "Save to Training Pool" button in header
      const btnSaveToPool = document.getElementById('btn-save-to-pool');
      if (btnSaveToPool) {
        btnSaveToPool.style.display = 'block';
      }

      // Load frames for key frame selection
      if (typeof loadFramesForSelection === 'function') {
        await loadFramesForSelection(WorkflowState.getCurrentUploadId(), WorkflowState.getCurrentVideoName());
      }

      // Update dashboard status to show step-3 as in-progress
      if (typeof updateDashboardStatus === 'function') {
        updateDashboardStatus();
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
