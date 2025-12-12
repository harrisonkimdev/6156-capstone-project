/**
 * Feedback Widget
 * 
 * Unified feedback display widget - fixed at bottom right of the screen.
 * Supports success, error, warning, and info message types.
 */

(function () {
  'use strict';

  // Widget container
  let container = null;
  const messageQueue = [];
  const maxVisibleMessages = 5;
  const defaultDuration = 5000; // 5 seconds

  /**
   * Initialize widget
   */
  function initWidget() {
    // Check if container already exists in DOM (prevents duplicate containers if script loads twice)
    const existingContainer = document.getElementById('feedback-widget-container');
    if (existingContainer) {
      container = existingContainer;
      console.log('[Feedback Widget] Using existing container from DOM');
      return;
    }

    // Check if container variable is already set
    if (container) {
      console.log('[Feedback Widget] Container already exists');
      return;
    }

    if (!document.body) {
      console.error('[Feedback Widget] document.body not available yet!');
      return;
    }

    container = document.createElement('div');
    container.id = 'feedback-widget-container';
    container.className = 'feedback-widget-container';
    document.body.appendChild(container);
    console.log('[Feedback Widget] Container created and added to DOM:', container, 'Body:', document.body);

    // Verify it's actually in the DOM
    const verify = document.getElementById('feedback-widget-container');
    if (verify) {
      console.log('[Feedback Widget] Container verified in DOM. Computed styles:', {
        display: window.getComputedStyle(verify).display,
        position: window.getComputedStyle(verify).position,
        bottom: window.getComputedStyle(verify).bottom,
        right: window.getComputedStyle(verify).right,
        zIndex: window.getComputedStyle(verify).zIndex,
        visibility: window.getComputedStyle(verify).visibility
      });
    } else {
      console.error('[Feedback Widget] Container NOT found in DOM after appendChild!');
    }
  }

  /**
   * Get icon for message type
   */
  function getIcon(type) {
    const icons = {
      success: '✓',
      error: '✗',
      warning: '⚠',
      info: 'ℹ'
    };
    return icons[type] || 'ℹ';
  }

  /**
   * Show feedback message
   * @param {string} message - Message to display
   * @param {string} type - Message type ('success', 'error', 'warning', 'info')
   * @param {number} duration - Auto-dismiss duration in ms (default 5000, 0 means no auto-dismiss)
   * @param {Array} actions - Action button array [{label: string, callback: function, style?: 'primary'|'secondary'}]
   */
  function showFeedback(message, type = 'info', duration = defaultDuration, actions = null) {
    if (!message) return;

    // Log to console
    const consoleMethod = type === 'error' ? 'error' : type === 'warning' ? 'warn' : 'log';
    const icon = getIcon(type);
    console[consoleMethod](`[Feedback ${icon}] ${message}`);

    // Initialize widget
    initWidget();

    // Debug: Check if container exists
    if (!container) {
      console.error('[Feedback Widget] Container not initialized!');
      return;
    }

    // Validate type
    const validTypes = ['success', 'error', 'warning', 'info'];
    if (!validTypes.includes(type)) {
      type = 'info';
    }

    // Set duration to 0 if actions present (keep until user clicks)
    // Otherwise use default duration (5 seconds) unless explicitly set
    if (actions && actions.length > 0 && duration === defaultDuration) {
      duration = 0;
    } else if (duration === undefined || duration === defaultDuration) {
      duration = defaultDuration;
    }

    // Create message item
    const item = document.createElement('div');
    item.className = `feedback-item ${type}`;
    if (actions && actions.length > 0) {
      item.classList.add('has-actions');
    }

    let actionsHtml = '';
    if (actions && actions.length > 0) {
      actionsHtml = `
        <div class="feedback-actions">
          ${actions.map((action, idx) => `
            <button class="feedback-action-btn ${action.style || (idx === 0 ? 'primary' : 'secondary')}" data-action-idx="${idx}">
              ${escapeHtml(action.label)}
            </button>
          `).join('')}
        </div>
      `;
    }

    // Add timer element if duration > 0
    const timerHtml = duration > 0 ? `<span class="feedback-timer">${Math.ceil(duration / 1000)}s</span>` : '';

    item.innerHTML = `
      <div class="feedback-content">
        <span class="feedback-icon">${icon}</span>
        <span class="feedback-message">${escapeHtml(message)}</span>
        ${timerHtml}
      </div>
      ${actionsHtml}
      <button class="feedback-close" aria-label="Close">×</button>
    `;

    // Close button event
    const closeBtn = item.querySelector('.feedback-close');
    closeBtn.addEventListener('click', () => {
      removeMessage(item);
    });

    // Action button events
    if (actions && actions.length > 0) {
      const actionBtns = item.querySelectorAll('.feedback-action-btn');
      actionBtns.forEach((btn) => {
        btn.addEventListener('click', () => {
          const idx = parseInt(btn.dataset.actionIdx);
          if (actions[idx] && typeof actions[idx].callback === 'function') {
            actions[idx].callback();
          }
          removeMessage(item);
        });
      });
    }

    // Add to container
    container.appendChild(item);
    console.log('[Feedback Widget] Item added to container:', item, 'Container:', container);

    // Small delay for animation
    requestAnimationFrame(() => {
      item.classList.add('show');
      console.log('[Feedback Widget] Show class added to item. Computed styles:', {
        display: window.getComputedStyle(item).display,
        opacity: window.getComputedStyle(item).opacity,
        transform: window.getComputedStyle(item).transform,
        visibility: window.getComputedStyle(item).visibility,
        zIndex: window.getComputedStyle(container).zIndex
      });
    });

    // Auto-remove with timer
    if (duration > 0) {
      let remainingTime = Math.ceil(duration / 1000);
      const timerElement = item.querySelector('.feedback-timer');

      // Update timer every second
      const intervalId = setInterval(() => {
        remainingTime--;
        if (timerElement) {
          if (remainingTime > 0) {
            timerElement.textContent = `${remainingTime}s`;
          } else {
            timerElement.textContent = '0s';
          }
        }

        if (remainingTime <= 0) {
          clearInterval(intervalId);
        }
      }, 1000);

      const timeoutId = setTimeout(() => {
        clearInterval(intervalId);
        removeMessage(item);
      }, duration);

      item.dataset.timeoutId = timeoutId;
      item.dataset.intervalId = intervalId;
    }

    // Queue management - remove oldest message if max exceeded
    messageQueue.push(item);
    if (messageQueue.length > maxVisibleMessages) {
      const oldestItem = messageQueue.shift();
      if (oldestItem && oldestItem.parentNode) {
        removeMessage(oldestItem);
      }
    }

    return item;
  }

  /**
   * Remove message
   */
  function removeMessage(item) {
    if (!item || !item.parentNode) return;

    // Clear timeout and interval
    if (item.dataset.timeoutId) {
      clearTimeout(parseInt(item.dataset.timeoutId));
    }
    if (item.dataset.intervalId) {
      clearInterval(parseInt(item.dataset.intervalId));
    }

    // Remove after animation
    item.classList.remove('show');
    item.classList.add('hide');

    setTimeout(() => {
      if (item.parentNode) {
        item.parentNode.removeChild(item);
      }
      // Remove from queue
      const index = messageQueue.indexOf(item);
      if (index > -1) {
        messageQueue.splice(index, 1);
      }
    }, 300); // Match CSS transition duration
  }

  /**
   * Escape HTML
   */
  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  /**
   * Clear all messages
   */
  function clearAll() {
    if (!container) return;

    const items = container.querySelectorAll('.feedback-item');
    items.forEach(item => removeMessage(item));
  }

  // Export as global functions
  window.showFeedback = showFeedback;
  window.clearFeedback = clearAll;

  // Initialize widget immediately if DOM is ready, otherwise wait for DOMContentLoaded
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      initWidget();
      console.log('[Feedback Widget] Initialized on DOMContentLoaded');
    });
  } else {
    initWidget();
    console.log('[Feedback Widget] Initialized immediately');
  }

  console.log('[Feedback Widget] Script loaded, showFeedback available:', typeof window.showFeedback === 'function');
})();
