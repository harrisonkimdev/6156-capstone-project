/**
 * Workflow Main Module
 * 
 * Initializes and coordinates all workflow modules.
 * This file loads all modules and sets up event listeners.
 */

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  setupEventListeners();

  // Initially hide step 2, 4 - they will be shown after "Start Analyzing"
  // Step 3 (frame selection) is shown for demo purposes
  const frameSelectionUI = document.getElementById('frame-selection-ui');
  const step2 = document.getElementById('step-2');
  const step3 = document.getElementById('step-3');
  const step4 = document.getElementById('step-4');

  // Initially show step 1 (upload video)
  if (step2) step2.style.display = 'none';
  if (step3) step3.style.display = 'none';
  if (step4) step4.style.display = 'none';
  if (frameSelectionUI) frameSelectionUI.style.display = 'none';

  // Set step 1 as active initially
  setStepActive('step-1');

  // Setup dashboard
  setupDashboard();

  // Setup training pool toggle
  setupTrainingPoolToggle();

  // Load initial data
  if (typeof loadTrainingJobs === 'function') {
    loadTrainingJobs();
  }
  if (typeof loadHoldDetectionPool === 'function') {
    loadHoldDetectionPool();
  }
  if (typeof loadSummaryInfo === 'function') {
    loadSummaryInfo();
  }

  // Poll for updates every 5 seconds
  setInterval(() => {
    if (typeof loadTrainingJobs === 'function') {
      loadTrainingJobs();
    }
    if (typeof updateDashboardStatus === 'function') {
      updateDashboardStatus();
    }
    if (typeof loadSummaryInfo === 'function') {
      loadSummaryInfo();
    }
    // Load training pools if visible
    const trainingPoolContent = document.getElementById('training-pool-jobs-content');
    if (trainingPoolContent && trainingPoolContent.style.display !== 'none') {
      const poolHoldsContent = document.getElementById('pool-holds-content');
      const poolFramesContent = document.getElementById('pool-frames-content');
      if (poolHoldsContent && poolHoldsContent.style.display !== 'none') {
        if (typeof loadHoldDetectionPool === 'function') {
          loadHoldDetectionPool();
        }
      }
      if (poolFramesContent && poolFramesContent.style.display !== 'none') {
        if (typeof loadKeyFrameSelectionPool === 'function') {
          loadKeyFrameSelectionPool();
        }
      }
    }
  }, 5000);
});

/**
 * Setup event listeners
 */
function setupEventListeners() {
  // Video file selection - show first frame preview
  const videoFileInput = document.getElementById('video-file');
  if (videoFileInput) {
    videoFileInput.addEventListener('change', function (event) {
      if (typeof handleVideoFileSelection === 'function') {
        handleVideoFileSelection(event);
      }
    });
  }

  // Hold color and route difficulty dropdown changes
  const holdColorSelect = document.getElementById('hold-color');
  if (holdColorSelect) {
    holdColorSelect.addEventListener('change', () => {
      if (typeof updateVideoPreview === 'function') {
        updateVideoPreview();
      }
    });
  }

  const routeDifficultySelect = document.getElementById('route-difficulty');
  if (routeDifficultySelect) {
    routeDifficultySelect.addEventListener('change', () => {
      if (typeof updateVideoPreview === 'function') {
        updateVideoPreview();
      }
    });
  }

  // Workflow step buttons
  const btnExtractFrames = document.getElementById('btn-extract-frames');
  if (btnExtractFrames) {
    btnExtractFrames.addEventListener('click', () => {
      if (typeof startAnalyzing === 'function') {
        startAnalyzing();
      }
    });
  }

  const btnCreateSession = document.getElementById('btn-create-session');
  if (btnCreateSession) {
    btnCreateSession.addEventListener('click', () => {
      if (typeof createSession === 'function') {
        createSession();
      }
    });
  }

  const btnStartTraining = document.getElementById('btn-start-training');
  if (btnStartTraining) {
    btnStartTraining.addEventListener('click', () => {
      if (typeof startTraining === 'function') {
        startTraining();
      }
    });
  }

  const btnUploadGCS = document.getElementById('btn-upload-gcs');
  if (btnUploadGCS) {
    btnUploadGCS.addEventListener('click', () => {
      if (typeof uploadToGCS === 'function') {
        uploadToGCS();
      }
    });
  }

  const btnClearData = document.getElementById('btn-clear-data');
  if (btnClearData) {
    btnClearData.addEventListener('click', () => {
      if (typeof clearAllData === 'function') {
        clearAllData();
      }
    });
  }

  // Frame selection UI
  const frameSlider = document.getElementById('frame-slider');
  if (frameSlider) {
    frameSlider.addEventListener('input', (event) => {
      if (typeof handleFrameSliderChange === 'function') {
        handleFrameSliderChange(event);
      }
    });
  }

  const btnSaveToPool = document.getElementById('btn-save-to-pool');
  if (btnSaveToPool) {
    btnSaveToPool.addEventListener('click', () => {
      if (typeof saveToTrainingPool === 'function') {
        saveToTrainingPool();
      }
    });
  }

  const btnTrainFrameSelector = document.getElementById('btn-train-frame-selector');
  if (btnTrainFrameSelector) {
    btnTrainFrameSelector.addEventListener('click', () => {
      if (typeof trainFrameSelector === 'function') {
        trainFrameSelector();
      }
    });
  }

  const btnViewAll = document.getElementById('btn-view-all');
  if (btnViewAll) {
    btnViewAll.addEventListener('click', () => {
      if (typeof setViewMode === 'function') {
        setViewMode('all');
      }
    });
  }

  const btnViewSelected = document.getElementById('btn-view-selected');
  if (btnViewSelected) {
    btnViewSelected.addEventListener('click', () => {
      if (typeof setViewMode === 'function') {
        setViewMode('selected');
      }
    });
  }

  // Keyboard shortcuts for frame selection
  document.addEventListener('keydown', (event) => {
    if (typeof handleKeyboardShortcuts === 'function') {
      handleKeyboardShortcuts(event);
    }
  });

  // Load training pool info on page load
  if (typeof loadTrainingPoolInfo === 'function') {
    loadTrainingPoolInfo();
  }

  // Hold labeling submit button
  const btnSubmitHolds = document.getElementById('btn-submit-holds');
  if (btnSubmitHolds) {
    btnSubmitHolds.addEventListener('click', () => {
      if (typeof submitHoldLabels === 'function') {
        submitHoldLabels();
      }
    });
  }

  // Training pool toggle buttons
  const btnPoolToggleHolds = document.getElementById('btn-pool-toggle-holds');
  const btnPoolToggleFrames = document.getElementById('btn-pool-toggle-frames');
  if (btnPoolToggleHolds) {
    btnPoolToggleHolds.addEventListener('click', () => {
      if (typeof toggleTrainingPool === 'function') {
        toggleTrainingPool('holds');
      }
    });
  }
  if (btnPoolToggleFrames) {
    btnPoolToggleFrames.addEventListener('click', () => {
      if (typeof toggleTrainingPool === 'function') {
        toggleTrainingPool('frames');
      }
    });
  }

  // Training pool train buttons
  const btnTrainYoloFromPool = document.getElementById('btn-train-yolo-from-pool');
  const btnTrainFrameSelectorFromPool = document.getElementById('btn-train-frame-selector-from-pool');
  if (btnTrainYoloFromPool) {
    btnTrainYoloFromPool.addEventListener('click', () => {
      if (typeof trainYoloFromPool === 'function') {
        trainYoloFromPool();
      }
    });
  }
  if (btnTrainFrameSelectorFromPool) {
    btnTrainFrameSelectorFromPool.addEventListener('click', () => {
      if (typeof trainFrameSelectorFromPool === 'function') {
        trainFrameSelectorFromPool();
      }
    });
  }
}

/**
 * Setup training pool toggle
 */
function setupTrainingPoolToggle() {
  const toggle = document.getElementById('toggle-training-pool');
  const content = document.getElementById('training-pool-jobs-content');

  if (!toggle || !content) {
    console.warn('Training pool toggle elements not found');
    return;
  }

  // Find elements within the toggle switch container
  const toggleSwitch = toggle.closest('.toggle-switch');
  if (!toggleSwitch) {
    console.warn('Toggle switch container not found');
    return;
  }

  const slider = toggleSwitch.querySelector('.toggle-slider');
  const knob = toggleSwitch.querySelector('.toggle-knob');
  const label = toggleSwitch.querySelector('#toggle-label');

  if (!slider || !knob || !label) {
    console.warn('Toggle switch sub-elements not found');
    return;
  }

  // Update toggle visual state
  function updateToggleState(isChecked) {
    if (isChecked) {
      slider.style.background = '#0066cc';
      slider.style.borderColor = '#0066cc';
      slider.style.boxShadow = 'inset 0 2px 4px rgba(0,0,0,0.2), 0 0 0 3px rgba(0,102,204,0.2)';
      knob.style.background = '#fff';
      knob.style.transform = 'translateX(28px)';
      knob.style.boxShadow = '0 2px 6px rgba(0,0,0,0.4), 0 0 0 0 rgba(0,102,204,0.3)';
      label.textContent = 'Hide';
      label.style.color = '#0066cc';
      content.style.display = 'block';
      // Load data when shown
      if (typeof loadHoldDetectionPool === 'function') {
        loadHoldDetectionPool();
      }
      if (typeof loadKeyFrameSelectionPool === 'function') {
        loadKeyFrameSelectionPool();
      }
      if (typeof loadTrainingJobs === 'function') {
        loadTrainingJobs();
      }
    } else {
      slider.style.background = '#333';
      slider.style.borderColor = '#444';
      slider.style.boxShadow = 'inset 0 2px 4px rgba(0,0,0,0.2)';
      knob.style.background = '#aaa';
      knob.style.transform = 'translateX(0)';
      knob.style.boxShadow = '0 2px 6px rgba(0,0,0,0.4)';
      label.textContent = 'Show';
      label.style.color = '#aaa';
      content.style.display = 'none';
    }
  }

  // Initial state - default to showing (since content is now visible by default)
  toggle.checked = true;
  updateToggleState(true);

  // Handle toggle change
  toggle.addEventListener('change', (e) => {
    updateToggleState(e.target.checked);
  });

  // Also handle click on slider (for better UX)
  slider.addEventListener('click', (e) => {
    e.stopPropagation();
    toggle.checked = !toggle.checked;
    toggle.dispatchEvent(new Event('change'));
  });

  // Handle click on label
  label.addEventListener('click', (e) => {
    e.stopPropagation();
    toggle.checked = !toggle.checked;
    toggle.dispatchEvent(new Event('change'));
  });

  // Add hover effects
  toggleSwitch.addEventListener('mouseenter', () => {
    if (!toggle.checked) {
      slider.style.borderColor = '#555';
      slider.style.boxShadow = 'inset 0 2px 4px rgba(0,0,0,0.2), 0 0 0 2px rgba(255,255,255,0.1)';
    } else {
      slider.style.boxShadow = 'inset 0 2px 4px rgba(0,0,0,0.2), 0 0 0 3px rgba(0,102,204,0.3)';
    }
    label.style.color = toggle.checked ? '#0088ff' : '#ccc';
  });

  toggleSwitch.addEventListener('mouseleave', () => {
    updateToggleState(toggle.checked);
  });
}
