/**
 * Training Pool Module
 * 
 * Manages training pool functionality for hold detection and key frame selection.
 */

/**
 * Get hold color label with emoji
 */
function getHoldColorLabel(color) {
  const colorMap = {
    'red': '🔴 Red',
    'blue': '🔵 Blue',
    'green': '🟢 Green',
    'yellow': '🟡 Yellow',
    'purple': '🟣 Purple',
    'black': '⚫ Black',
    'white': '⚪ White',
  };
  return colorMap[color] || color;
}

/**
 * Get route difficulty label with emoji
 */
function getRouteDifficultyLabel(difficulty) {
  const difficultyMap = {
    'pink': '🟪 Pink (Easier)',
    'black': '⬛ Black',
    'yellow': '🟨 Yellow',
    'orange': '🟧 Orange',
    'white': '⬜ White',
    'purple': '🟪 Purple',
    'green': '🟩 Green',
    'blue': '🟦 Blue (Harder)',
  };
  return difficultyMap[difficulty] || difficulty;
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

    const poolVideoCount = document.getElementById('pool-video-count');
    const poolFrameCount = document.getElementById('pool-frame-count');

    if (poolVideoCount) {
      poolVideoCount.textContent = data.video_count || 0;
    }
    if (poolFrameCount) {
      poolFrameCount.textContent = data.frame_count || 0;
    }

    // Also update header counts
    updatePoolHeaderCounts();

  } catch (error) {
    console.error('Failed to load pool info:', error);
  }
}

/**
 * Toggle between Hold Detection Labels and Key Frame Selection Labels pools
 * @param {string} poolType - Pool type ('holds' or 'frames')
 */
function toggleTrainingPool(poolType) {
  const btnHolds = document.getElementById('btn-pool-toggle-holds');
  const btnFrames = document.getElementById('btn-pool-toggle-frames');
  const contentHolds = document.getElementById('pool-holds-content');
  const contentFrames = document.getElementById('pool-frames-content');

  if (poolType === 'holds') {
    if (btnHolds) {
      btnHolds.style.background = '#0066cc';
      btnHolds.style.color = 'white';
    }
    if (btnFrames) {
      btnFrames.style.background = '#333';
      btnFrames.style.color = '#aaa';
    }
    if (contentHolds) {
      contentHolds.style.display = 'block';
    }
    if (contentFrames) {
      contentFrames.style.display = 'none';
    }
    loadHoldDetectionPool();
  } else {
    if (btnHolds) {
      btnHolds.style.background = '#333';
      btnHolds.style.color = '#aaa';
    }
    if (btnFrames) {
      btnFrames.style.background = '#0066cc';
      btnFrames.style.color = 'white';
    }
    if (contentHolds) {
      contentHolds.style.display = 'none';
    }
    if (contentFrames) {
      contentFrames.style.display = 'block';
    }
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
    const poolHoldsCount = document.getElementById('pool-holds-count');
    if (poolHoldsCount) {
      poolHoldsCount.textContent = data.total_sets || 0;
    }

    const listContainer = document.getElementById('pool-holds-list');
    if (!listContainer) return;

    if (data.sets && data.sets.length > 0) {
      listContainer.innerHTML = data.sets.map(set => {
        const holdColorLabel = set.hold_color ? getHoldColorLabel(set.hold_color) : null;
        const routeDifficultyLabel = set.route_difficulty ? getRouteDifficultyLabel(set.route_difficulty) : null;

        return `
        <div style="display: flex; gap: 12px; padding: 10px; margin-bottom: 8px; background: #1a1a1a; border-radius: 4px; border-left: 3px solid #0066cc;">
          ${set.frame_image_url ? `
            <img src="${set.frame_image_url}" 
                 alt="First frame" 
                 style="width: 120px; height: 90px; object-fit: cover; border-radius: 4px; flex-shrink: 0;" />
          ` : ''}
          <div style="flex: 1; min-width: 0;">
            <div style="color: #fff; font-weight: bold; margin-bottom: 5px;">${set.name || set.id}</div>
            <div style="color: #aaa; font-size: 12px; margin-bottom: 4px;">
              ${set.labeled_segments || 0} labeled segments | ${set.created_at || ''}
            </div>
            ${holdColorLabel || routeDifficultyLabel ? `
              <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 4px;">
                ${holdColorLabel ? `<span style="color: #aaa; font-size: 11px;">${holdColorLabel}</span>` : ''}
                ${routeDifficultyLabel ? `<span style="color: #aaa; font-size: 11px;">${routeDifficultyLabel}</span>` : ''}
              </div>
            ` : ''}
          </div>
        </div>
      `;
      }).join('');
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
    const poolVideoCount = document.getElementById('pool-video-count');
    const poolFrameCount = document.getElementById('pool-frame-count');

    if (poolVideoCount) {
      poolVideoCount.textContent = data.video_count || 0;
    }
    if (poolFrameCount) {
      poolFrameCount.textContent = data.frame_count || 0;
    }

    const listContainer = document.getElementById('pool-frames-list');
    if (!listContainer) return;

    if (data.videos && data.videos.length > 0) {
      listContainer.innerHTML = data.videos.map(video => {
        const holdColorLabel = video.hold_color ? getHoldColorLabel(video.hold_color) : null;
        const routeDifficultyLabel = video.route_difficulty ? getRouteDifficultyLabel(video.route_difficulty) : null;

        return `
        <div style="display: flex; gap: 12px; padding: 10px; margin-bottom: 8px; background: #1a1a1a; border-radius: 4px; border-left: 3px solid #0066cc;">
          ${video.frame_image_url ? `
            <img src="${video.frame_image_url}" 
                 alt="First frame" 
                 style="width: 120px; height: 90px; object-fit: cover; border-radius: 4px; flex-shrink: 0;" />
          ` : ''}
          <div style="flex: 1; min-width: 0;">
            <div style="color: #fff; font-weight: bold; margin-bottom: 5px;">${video.name || video.id}</div>
            <div style="color: #aaa; font-size: 12px; margin-bottom: 4px;">
              ${video.selected_frames || 0} selected frames | ${video.created_at || ''}
            </div>
            ${holdColorLabel || routeDifficultyLabel ? `
              <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 4px;">
                ${holdColorLabel ? `<span style="color: #aaa; font-size: 11px;">${holdColorLabel}</span>` : ''}
                ${routeDifficultyLabel ? `<span style="color: #aaa; font-size: 11px;">${routeDifficultyLabel}</span>` : ''}
              </div>
            ` : ''}
          </div>
        </div>
      `;
      }).join('');
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

    if (typeof loadTrainingJobs === 'function') {
      loadTrainingJobs();
    }

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
 * Highlight training pool section to show where data was added
 * @param {string} poolType - Pool type ('holds' or 'frames')
 */
function highlightPoolSection(poolType) {
  const trainingPoolSection = document.getElementById('training-pool-section');
  if (!trainingPoolSection) return;

  // First switch to the relevant pool tab
  toggleTrainingPool(poolType);

  // Add highlight class
  trainingPoolSection.classList.add('pool-highlight');

  // Remove highlight after animation completes
  setTimeout(() => {
    trainingPoolSection.classList.remove('pool-highlight');
  }, 3000);

  // Scroll to the training pool section
  trainingPoolSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

/**
 * Update pool counts in the header
 */
async function updatePoolHeaderCounts() {
  try {
    // Get hold detection pool count
    const holdsResponse = await fetch('/api/workflow/pool/hold-detection');
    if (holdsResponse.ok) {
      const holdsData = await holdsResponse.json();
      const holdCountBadge = document.getElementById('pool-header-holds-count');
      if (holdCountBadge) {
        holdCountBadge.textContent = holdsData.total_sets || 0;
      }
    }

    // Get key frame selection pool count
    const framesResponse = await fetch('/api/workflow/pool/key-frame-selection');
    if (framesResponse.ok) {
      const framesData = await framesResponse.json();
      const frameCountBadge = document.getElementById('pool-header-frames-count');
      if (frameCountBadge) {
        frameCountBadge.textContent = framesData.video_count || 0;
      }
    }
  } catch (error) {
    console.error('Failed to update pool header counts:', error);
  }
}
