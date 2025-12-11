/**
 * Dashboard Module
 * 
 * Manages workflow dashboard and progress bar functionality.
 */

/**
 * Setup dashboard progress bar click handlers
 */
function setupDashboard() {
  const workflowSteps = document.querySelectorAll('.workflow-step-card');
  workflowSteps.forEach(step => {
    step.addEventListener('click', () => {
      const stepId = step.dataset.step;
      navigateToStep(stepId);
    });
  });

  // Initial dashboard status update
  updateDashboardStatus();
}

/**
 * Navigate to a specific step
 * @param {string} stepId - Step ID
 */
function navigateToStep(stepId) {
  // Hide all steps first
  const steps = ['step-1', 'step-2', 'step-3', 'step-4'];
  steps.forEach(id => {
    const step = document.getElementById(id);
    if (step) {
      step.style.display = 'none';
    }
  });

  // Show only the selected step
  const targetStep = document.getElementById(stepId);
  if (targetStep) {
    targetStep.style.display = 'block';
    setStepActive(stepId);

    // Scroll to step
    setTimeout(() => {
      targetStep.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
  }

  // Update dashboard active state
  updateDashboardActiveState(stepId);
  updateDashboardStatus();
}

/**
 * Get step status: 'completed', 'in-progress', or 'not-started'
 * @param {string} stepId - Step ID
 * @returns {string} Status
 */
function getStepStatus(stepId) {
  switch (stepId) {
    case 'step-1':
      // 비디오가 업로드되어 있으면 completed
      if (WorkflowState.getCurrentUploadId() && WorkflowState.getCurrentVideoName()) {
        return 'completed';
      }
      // 비디오 파일이 선택되어 있으면 in-progress (업로드/분석 진행 중)
      const videoFileInput = document.getElementById('video-file');
      if (videoFileInput && videoFileInput.files && videoFileInput.files.length > 0) {
        return 'in-progress';
      }
      return 'not-started';
    case 'step-2':
      // If labels were submitted, step-2 is completed
      if (WorkflowState.getHoldLabelsSubmitted && WorkflowState.getHoldLabelsSubmitted()) {
        return 'completed';
      }

      const segments = WorkflowState.getHoldLabelingSegments();
      // segments가 없으면 not-started
      if (segments.length === 0) {
        const labelingUI = document.getElementById('hold-labeling-ui');
        if (labelingUI && labelingUI.style.display !== 'none' &&
          WorkflowState.getCurrentUploadId() && WorkflowState.getCurrentVideoName()) {
          return 'in-progress'; // SAM segmentation 진행 중
        }
        return 'not-started';
      }

      // segments가 있으면 모든 segments에 hold_type이 설정되었는지 확인
      const allLabeled = segments.every(seg => seg.hold_type && seg.hold_type !== '');
      if (allLabeled) {
        return 'completed';
      }

      // segments가 있지만 아직 labeling이 완료되지 않음
      return 'in-progress';
    case 'step-3':
      // Step 3 is completed if sessionId exists
      // In-progress only if video is uploaded and frame selection UI is active
      if (WorkflowState.getCurrentSessionId()) {
        return 'completed';
      }
      // Only in-progress if step-1 is completed and frame selection UI is visible (actual work)
      if (WorkflowState.getCurrentUploadId() && WorkflowState.getCurrentVideoName()) {
        const frameSelectionUI = document.getElementById('frame-selection-ui');
        if (frameSelectionUI && frameSelectionUI.style.display !== 'none') {
          return 'in-progress';
        }
      }
      return 'not-started';
    case 'step-4':
      // Step 4 is in-progress if training job exists, otherwise not-started
      if (WorkflowState.getCurrentTrainingJobId()) {
        return 'in-progress';
      }
      return 'not-started';
    default:
      return 'not-started';
  }
}

/**
 * Update dashboard active state (progress bar)
 * @param {string} activeStepId - Active step ID
 */
function updateDashboardActiveState(activeStepId) {
  const workflowSteps = document.querySelectorAll('.workflow-step-card');
  workflowSteps.forEach(step => {
    const stepId = step.dataset.step;
    const status = getStepStatus(stepId);
    const isActive = stepId === activeStepId;

    const indicator = step.querySelector('.step-indicator');
    const number = step.querySelector('.step-number');
    const icon = step.querySelector('.step-icon');

    // Reset classes
    step.classList.remove('active', 'completed', 'in-progress', 'not-started', 'error');
    if (indicator) {
      indicator.classList.remove('active', 'completed', 'in-progress', 'not-started');
    }

    if (isActive) {
      step.classList.add('active');
      if (indicator) indicator.classList.add('active');
      if (number) number.style.display = 'none';
      if (icon) icon.style.display = 'flex';
    } else {
      if (status === 'completed') {
        step.classList.add('completed');
        if (indicator) indicator.classList.add('completed');
        if (number) number.style.display = 'none';
        if (icon) {
          icon.style.display = 'flex';
          icon.textContent = '✓';
        }
      } else if (status === 'in-progress') {
        step.classList.add('in-progress');
        if (indicator) indicator.classList.add('in-progress');
        if (number) number.style.display = 'none';
        if (icon) {
          icon.style.display = 'flex';
          icon.textContent = '⟳';
        }
      } else {
        step.classList.add('not-started');
        if (indicator) indicator.classList.add('not-started');
        if (number) number.style.display = 'flex';
        if (icon) icon.style.display = 'none';
      }
    }
  });
}

/**
 * Check if a step has been started
 * @param {string} stepId - Step ID
 * @returns {boolean} Whether step has been started
 */
function isStepStarted(stepId) {
  switch (stepId) {
    case 'step-1':
      return WorkflowState.getCurrentUploadId() && WorkflowState.getCurrentVideoName();
    case 'step-2':
      return WorkflowState.getHoldLabelingSegments().length > 0;
    case 'step-3':
      return WorkflowState.getCurrentSessionId() !== null;
    case 'step-4':
      return WorkflowState.getCurrentTrainingJobId() !== null;
    default:
      return false;
  }
}

/**
 * Update dashboard status for each step (progress bar)
 */
function updateDashboardStatus() {
  const steps = ['step-1', 'step-2', 'step-3', 'step-4'];
  let completedCount = 0;
  let activeStep = null;

  // Update each step status badge
  steps.forEach((stepId) => {
    const status = getStepStatus(stepId);
    const statusBadge = document.getElementById(`${stepId}-status-badge`);

    if (status === 'completed') {
      completedCount++;
      if (statusBadge) {
        statusBadge.textContent = 'Completed';
        statusBadge.className = 'step-status completed';
      }
    } else if (status === 'in-progress') {
      if (statusBadge) {
        statusBadge.textContent = 'In progress';
        statusBadge.className = 'step-status in-progress';
      }
      if (!activeStep) {
        activeStep = stepId;
      }
    } else {
      if (statusBadge) {
        statusBadge.textContent = 'Not started';
        statusBadge.className = 'step-status not-started';
      }
    }

    // Check if step is currently visible
    const step = document.getElementById(stepId);
    if (step && step.style.display !== 'none' && !activeStep) {
      activeStep = stepId;
    }
  });

  // Update progress text
  const progressText = document.getElementById('workflow-progress-text');
  if (progressText) {
    const percentage = Math.round((completedCount / steps.length) * 100);
    progressText.textContent = `${completedCount}/${steps.length} steps completed (${percentage}%)`;
  }

  // Update active state
  if (activeStep) {
    updateDashboardActiveState(activeStep);
  } else {
    // If no active step, find the first not-started step
    for (const stepId of steps) {
      const status = getStepStatus(stepId);
      if (status === 'not-started') {
        updateDashboardActiveState(stepId);
        break;
      }
    }
  }
}
