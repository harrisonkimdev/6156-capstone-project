/**
 * Feedback Widget
 * 
 * 통일된 피드백 표시 위젯 - 오른쪽 하단에 고정되어 표시됩니다.
 * success, error, warning, info 타입을 지원합니다.
 */

(function() {
  'use strict';

  // 위젯 컨테이너 생성
  let container = null;
  const messageQueue = [];
  const maxVisibleMessages = 5;
  const defaultDuration = 5000; // 5초

  /**
   * 위젯 초기화
   */
  function initWidget() {
    if (container) return;

    container = document.createElement('div');
    container.id = 'feedback-widget-container';
    container.className = 'feedback-widget-container';
    document.body.appendChild(container);
  }

  /**
   * 타입별 아이콘 반환
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
   * 피드백 메시지 표시
   * @param {string} message - 표시할 메시지
   * @param {string} type - 메시지 타입 ('success', 'error', 'warning', 'info')
   * @param {number} duration - 자동으로 사라지는 시간 (ms, 기본 5000)
   */
  function showFeedback(message, type = 'info', duration = defaultDuration) {
    if (!message) return;

    // 위젯 초기화
    initWidget();

    // 타입 검증
    const validTypes = ['success', 'error', 'warning', 'info'];
    if (!validTypes.includes(type)) {
      type = 'info';
    }

    // 메시지 아이템 생성
    const item = document.createElement('div');
    item.className = `feedback-item ${type}`;
    
    const icon = getIcon(type);
    item.innerHTML = `
      <span class="feedback-icon">${icon}</span>
      <span class="feedback-message">${escapeHtml(message)}</span>
      <button class="feedback-close" aria-label="Close">×</button>
    `;

    // 닫기 버튼 이벤트
    const closeBtn = item.querySelector('.feedback-close');
    closeBtn.addEventListener('click', () => {
      removeMessage(item);
    });

    // 컨테이너에 추가
    container.appendChild(item);

    // 애니메이션을 위한 약간의 지연
    requestAnimationFrame(() => {
      item.classList.add('show');
    });

    // 자동 제거
    if (duration > 0) {
      const timeoutId = setTimeout(() => {
        removeMessage(item);
      }, duration);
      item.dataset.timeoutId = timeoutId;
    }

    // 큐 관리 - 최대 개수 초과 시 가장 오래된 메시지 제거
    messageQueue.push(item);
    if (messageQueue.length > maxVisibleMessages) {
      const oldestItem = messageQueue.shift();
      if (oldestItem && oldestItem.parentNode) {
        removeMessage(oldestItem);
      }
    }
  }

  /**
   * 메시지 제거
   */
  function removeMessage(item) {
    if (!item || !item.parentNode) return;

    // 타임아웃 클리어
    if (item.dataset.timeoutId) {
      clearTimeout(parseInt(item.dataset.timeoutId));
    }

    // 애니메이션 후 제거
    item.classList.remove('show');
    item.classList.add('hide');

    setTimeout(() => {
      if (item.parentNode) {
        item.parentNode.removeChild(item);
      }
      // 큐에서도 제거
      const index = messageQueue.indexOf(item);
      if (index > -1) {
        messageQueue.splice(index, 1);
      }
    }, 300); // CSS transition 시간과 일치
  }

  /**
   * HTML 이스케이프
   */
  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  /**
   * 모든 메시지 제거
   */
  function clearAll() {
    if (!container) return;
    
    const items = container.querySelectorAll('.feedback-item');
    items.forEach(item => removeMessage(item));
  }

  // 전역 함수로 export
  window.showFeedback = showFeedback;
  window.clearFeedback = clearAll;

  // DOMContentLoaded 시 초기화
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initWidget);
  } else {
    initWidget();
  }
})();
