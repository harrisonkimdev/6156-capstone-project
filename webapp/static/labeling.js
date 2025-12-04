/**
 * Hold Labeling UI - Frontend JavaScript
 * 
 * Manages canvas rendering, segment selection, and labeling workflow.
 */

// Hold types mapping (matches backend HOLD_TYPE_TO_CLASS_ID)
const HOLD_TYPES = {
  '': 'Not a hold',
  'crimp': 'Crimp',
  'sloper': 'Sloper',
  'jug': 'Jug',
  'pinch': 'Pinch',
  'foot_only': 'Foot Only',
  'volume': 'Volume',
};

// State
let sessionId = null;
let currentFrameIndex = 0;
let frames = [];
let currentSegments = [];
let selectedSegmentId = null;
let canvas = null;
let ctx = null;
let currentImage = null;

// Color palette for segments
const SEGMENT_COLORS = [
  '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
  '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B88B', '#AAB7B8',
];

/**
 * Initialize the labeling UI
 */
async function init() {
  // Get session ID from URL query params
  const urlParams = new URLSearchParams(window.location.search);
  sessionId = urlParams.get('session_id');

  if (!sessionId) {
    showStatus('No session ID provided', 'error');
    return;
  }

  // Initialize canvas
  canvas = document.getElementById('image-canvas');
  ctx = canvas.getContext('2d');

  // Set up event listeners
  setupEventListeners();

  // Load session data
  await loadSession();
}

/**
 * Set up event listeners
 */
function setupEventListeners() {
  // Frame navigation
  document.getElementById('prev-frame').addEventListener('click', () => navigateFrame(-1));
  document.getElementById('next-frame').addEventListener('click', () => navigateFrame(1));

  // Action buttons
  document.getElementById('save-labels').addEventListener('click', saveLabels);
  document.getElementById('export-dataset').addEventListener('click', exportDataset);
  document.getElementById('clear-frame').addEventListener('click', clearFrame);

  // Canvas click for segment selection
  canvas.addEventListener('click', handleCanvasClick);

  // Keyboard shortcuts
  document.addEventListener('keydown', handleKeyPress);
}

/**
 * Load session data from backend
 */
async function loadSession() {
  try {
    showStatus('Loading session...', 'info');

    const response = await fetch(`/api/labeling/sessions/${sessionId}`);
    if (!response.ok) {
      throw new Error(`Failed to load session: ${response.statusText}`);
    }

    const data = await response.json();
    frames = data.frames;

    showStatus(`Loaded ${frames.length} frames`, 'success');
    updateProgress(data.progress);

    // Load first frame
    if (frames.length > 0) {
      await loadFrame(0);
    }

  } catch (error) {
    console.error('Failed to load session:', error);
    showStatus(`Error: ${error.message}`, 'error');
  }
}

/**
 * Load a specific frame
 */
async function loadFrame(frameIndex) {
  if (frameIndex < 0 || frameIndex >= frames.length) {
    return;
  }

  currentFrameIndex = frameIndex;

  try {
    showStatus('Loading frame...', 'info');

    const response = await fetch(`/api/labeling/sessions/${sessionId}/frames/${frameIndex}`);
    if (!response.ok) {
      throw new Error(`Failed to load frame: ${response.statusText}`);
    }

    const data = await response.json();
    currentSegments = data.segments;

    // Load image
    await loadImage(data.image_url);

    // Render frame
    renderFrame();

    // Update UI
    updateFrameCounter();
    renderSegmentList();

    showStatus('', 'info');

  } catch (error) {
    console.error('Failed to load frame:', error);
    showStatus(`Error: ${error.message}`, 'error');
  }
}

/**
 * Load image onto canvas
 */
function loadImage(imageUrl) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      currentImage = img;
      canvas.width = img.width;
      canvas.height = img.height;
      resolve();
    };
    img.onerror = reject;
    img.src = imageUrl;
  });
}

/**
 * Render frame with segments overlay
 */
function renderFrame() {
  if (!currentImage) return;

  // Clear canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw image
  ctx.drawImage(currentImage, 0, 0);

  // Draw segments
  currentSegments.forEach((segment, idx) => {
    const color = SEGMENT_COLORS[idx % SEGMENT_COLORS.length];
    const isSelected = segment.segment_id === selectedSegmentId;
    const isLabeled = segment.is_hold && segment.user_confirmed;

    drawSegment(segment, color, isSelected, isLabeled);
  });
}

/**
 * Draw a single segment
 */
function drawSegment(segment, color, isSelected, isLabeled) {
  const [x1, y1, x2, y2] = segment.bbox;

  // Draw bounding box
  ctx.strokeStyle = isSelected ? '#00FF00' : color;
  ctx.lineWidth = isSelected ? 6 : 3;
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

  // Draw filled rect for labeled segments
  if (isLabeled) {
    ctx.fillStyle = color + '40'; // 25% opacity
    ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
  }

  // Draw label
  if (isLabeled && segment.hold_type) {
    const labelText = HOLD_TYPES[segment.hold_type] || segment.hold_type;
    ctx.fillStyle = color;
    ctx.font = 'bold 14px Arial';
    ctx.fillText(labelText, x1 + 5, y1 + 20);
  }
}

/**
 * Render segment list in right panel
 */
function renderSegmentList() {
  const listEl = document.getElementById('segment-list');
  listEl.innerHTML = '';

  document.getElementById('segments-count').textContent = `${currentSegments.length} segments`;

  currentSegments.forEach((segment, idx) => {
    const card = createSegmentCard(segment, idx);
    listEl.appendChild(card);
  });
}

/**
 * Create a segment card element
 */
function createSegmentCard(segment, idx) {
  const card = document.createElement('div');
  card.className = 'segment-card';
  card.dataset.segmentId = segment.segment_id;

  if (segment.segment_id === selectedSegmentId) {
    card.classList.add('selected');
  }

  if (segment.is_hold && segment.user_confirmed) {
    card.classList.add('labeled');
  }

  // Segment info
  const info = document.createElement('div');
  info.className = 'segment-info';
  info.innerHTML = `
        <span class="segment-id">#${idx + 1}</span>
        <span class="segment-score">Score: ${segment.stability_score.toFixed(2)}</span>
    `;
  card.appendChild(info);

  // Hold type selector
  const selector = document.createElement('select');
  selector.className = 'hold-type-selector';

  for (const [value, label] of Object.entries(HOLD_TYPES)) {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = label;
    if (value === (segment.hold_type || '')) {
      option.selected = true;
    }
    selector.appendChild(option);
  }

  selector.addEventListener('change', (e) => {
    e.stopPropagation();
    e.preventDefault();
    updateSegmentLabel(segment.segment_id, e.target.value);
  });

  // Prevent dropdown from closing when clicking on it
  selector.addEventListener('click', (e) => {
    e.stopPropagation();
  });

  selector.addEventListener('mousedown', (e) => {
    e.stopPropagation();
  });

  card.appendChild(selector);

  // Click to select (only on card, not on selector)
  card.addEventListener('click', (e) => {
    // Don't select if clicking on the dropdown or its children (options)
    if (e.target === selector || selector.contains(e.target)) {
      return;
    }
    selectSegment(segment.segment_id);
  });

  return card;
}

/**
 * Handle canvas click for segment selection
 */
function handleCanvasClick(event) {
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;

  // Scale to canvas coordinates
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const canvasX = x * scaleX;
  const canvasY = y * scaleY;

  // Find clicked segment
  for (const segment of currentSegments) {
    const [x1, y1, x2, y2] = segment.bbox;
    if (canvasX >= x1 && canvasX <= x2 && canvasY >= y1 && canvasY <= y2) {
      selectSegment(segment.segment_id);
      return;
    }
  }

  // Deselect if clicked outside
  selectSegment(null);
}

/**
 * Select a segment
 */
function selectSegment(segmentId) {
  selectedSegmentId = segmentId;
  renderFrame();

  // Update card selected state without full re-render
  document.querySelectorAll('.segment-card').forEach(card => {
    card.classList.remove('selected');
  });

  if (segmentId) {
    const card = document.querySelector(`[data-segment-id="${segmentId}"]`);
    if (card) {
      card.classList.add('selected');
      card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }
}

/**
 * Update segment label
 */
async function updateSegmentLabel(segmentId, holdType) {
  const isHold = holdType !== '';

  try {
    const response = await fetch(`/api/labeling/sessions/${sessionId}/labels`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        frame_index: currentFrameIndex,
        segment_id: segmentId,
        hold_type: holdType || null,
        is_hold: isHold,
      }),
    });

    if (!response.ok) {
      throw new Error(`Failed to update label: ${response.statusText}`);
    }

    const data = await response.json();

    // Update local state
    const segment = currentSegments.find(s => s.segment_id === segmentId);
    if (segment) {
      segment.hold_type = holdType || null;
      segment.is_hold = isHold;
      segment.user_confirmed = true;
    }

    // Update progress
    updateProgress(data.progress);

    // Re-render only the canvas (not the segment list to preserve dropdown state)
    renderFrame();

    // Update the specific segment card styling without full re-render
    const card = document.querySelector(`.segment-card[data-segment-id="${segmentId}"]`);
    if (card && segment) {
      if (segment.is_hold && segment.user_confirmed) {
        card.classList.add('labeled');
      } else {
        card.classList.remove('labeled');
      }
    }

  } catch (error) {
    console.error('Failed to update label:', error);
    showStatus(`Error: ${error.message}`, 'error');
  }
}

/**
 * Navigate between frames
 */
function navigateFrame(delta) {
  const newIndex = currentFrameIndex + delta;
  if (newIndex >= 0 && newIndex < frames.length) {
    loadFrame(newIndex);
  }
}

/**
 * Update frame counter
 */
function updateFrameCounter() {
  document.getElementById('frame-counter').textContent =
    `Frame ${currentFrameIndex + 1} / ${frames.length}`;

  // Update button states
  document.getElementById('prev-frame').disabled = currentFrameIndex === 0;
  document.getElementById('next-frame').disabled = currentFrameIndex === frames.length - 1;
}

/**
 * Update progress bar
 */
function updateProgress(progress) {
  const percent = progress.completion_percent || 0;
  const labeledSegments = progress.labeled_segments || 0;
  const totalSegments = progress.total_segments || 0;

  document.getElementById('progress-fill').style.width = `${percent}%`;
  document.getElementById('progress-fill').textContent = `${percent}%`;
  document.getElementById('progress-text').textContent =
    `Labeled ${labeledSegments} / ${totalSegments} holds`;
}

/**
 * Save labels
 */
async function saveLabels() {
  try {
    showStatus('Saving labels...', 'info');

    const response = await fetch(`/api/labeling/sessions/${sessionId}/save`, {
      method: 'POST',
    });

    if (!response.ok) {
      throw new Error(`Failed to save: ${response.statusText}`);
    }

    showStatus('Labels saved successfully!', 'success');
    setTimeout(() => showStatus('', 'info'), 3000);

  } catch (error) {
    console.error('Failed to save labels:', error);
    showStatus(`Error: ${error.message}`, 'error');
  }
}

/**
 * Export dataset to YOLO format
 */
async function exportDataset() {
  if (!confirm('Export labeled data to YOLO training dataset?')) {
    return;
  }

  try {
    showStatus('Exporting dataset...', 'info');

    const response = await fetch(`/api/labeling/sessions/${sessionId}/export`, {
      method: 'POST',
    });

    if (!response.ok) {
      throw new Error(`Failed to export: ${response.statusText}`);
    }

    const data = await response.json();
    showStatus(`Exported ${data.exported_count} images to YOLO dataset!`, 'success');

  } catch (error) {
    console.error('Failed to export dataset:', error);
    showStatus(`Error: ${error.message}`, 'error');
  }
}

/**
 * Clear frame labels
 */
function clearFrame() {
  if (!confirm('Clear all labels for this frame?')) {
    return;
  }

  currentSegments.forEach(segment => {
    updateSegmentLabel(segment.segment_id, '');
  });
}

/**
 * Handle keyboard shortcuts
 */
function handleKeyPress(event) {
  // Ignore if typing in input
  if (event.target.tagName === 'SELECT' || event.target.tagName === 'INPUT') {
    return;
  }

  switch (event.key.toLowerCase()) {
    case 'a':
      navigateFrame(-1);
      break;
    case 'd':
      navigateFrame(1);
      break;
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
      if (selectedSegmentId) {
        const holdTypes = ['crimp', 'sloper', 'jug', 'pinch', 'foot_only', 'volume'];
        const holdType = holdTypes[parseInt(event.key) - 1];
        updateSegmentLabel(selectedSegmentId, holdType);
      }
      break;
    case 'escape':
      selectSegment(null);
      break;
  }
}

/**
 * Show status message
 */
function showStatus(message, type) {
  const el = document.getElementById('status-message');
  el.textContent = message;
  el.className = 'status-message ' + type;

  if (!message) {
    el.style.display = 'none';
  }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', init);
