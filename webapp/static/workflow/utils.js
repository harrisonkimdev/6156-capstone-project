/**
 * Workflow Utility Functions
 * 
 * Common utility functions used across workflow modules.
 */

/**
 * Show status message for a step
 * @param {string} stepId - Step ID (e.g., 'step-1')
 * @param {string} message - Status message
 * @param {string} type - Status type ('success', 'error', 'info')
 */
function showStatus(stepId, message, type) {
  const el = document.getElementById(`status-${stepId}`);
  if (el) {
    el.textContent = message;
    el.className = `step-status ${type}`;
    el.style.display = 'block';
  }
}

/**
 * Set step as active
 * @param {string} stepId - Step ID
 */
function setStepActive(stepId) {
  document.querySelectorAll('.step-card').forEach(card => {
    card.classList.remove('active');
  });
  const step = document.getElementById(stepId);
  if (step) {
    step.classList.add('active');
  }
}

/**
 * Set step as completed
 * @param {string} stepId - Step ID
 */
function setStepCompleted(stepId) {
  const step = document.getElementById(stepId);
  if (step) {
    step.classList.add('completed');
  }
}

/**
 * Scroll to a specific step smoothly
 * @param {string} stepId - Step ID
 */
function scrollToStep(stepId) {
  const stepElement = document.getElementById(stepId);
  if (stepElement) {
    stepElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

/**
 * Generate session name from current date/time
 * @returns {string} Session name in format YYYY-MM-DD_HHMM
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
