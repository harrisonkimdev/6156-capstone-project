/**
 * Job Notifications Module
 * 
 * Monitors for new analysis jobs created via external API and shows notifications.
 * Uses feedback widget for displaying notifications.
 */

(function () {
  'use strict';

  let knownJobIds = new Set();
  let isPolling = false;
  const POLL_INTERVAL = 5000; // 5 seconds

  /**
   * Check for new jobs and show notifications
   */
  async function checkForNewJobs() {
    if (isPolling) return;
    isPolling = true;

    try {
      const response = await fetch('/api/jobs');
      if (!response.ok) return;

      const jobs = await response.json();
      const currentJobIds = new Set(jobs.map(job => job.id));

      // Find new jobs (jobs that weren't in knownJobIds)
      const newJobs = jobs.filter(job => !knownJobIds.has(job.id));

      // Show notification for each new job
      newJobs.forEach(job => {
        showJobNotification(job);
      });

      // Update known job IDs
      knownJobIds = currentJobIds;

    } catch (error) {
      console.error('Failed to check for new jobs:', error);
    } finally {
      isPolling = false;
    }
  }

  /**
   * Show notification for a new job
   */
  function showJobNotification(job) {
    if (!window.showFeedback) {
      console.warn('Feedback widget not available');
      return;
    }

    // Build notification message with detailed info
    const jobIdShort = job.id.substring(0, 8);
    const message = `New analysis job: ${jobIdShort}`;

    // Extract metadata details
    const metadata = job.metadata || {};
    const details = [];

    details.push(`Status: ${job.status}`);

    if (metadata.route_name) {
      details.push(`Route: ${metadata.route_name}`);
    }

    if (metadata.gym_location) {
      details.push(`Location: ${metadata.gym_location}`);
    }

    if (job.video_dir) {
      const videoName = job.video_dir.split('/').pop();
      details.push(`Video: ${videoName}`);
    }

    if (metadata.climber_height) {
      details.push(`Height: ${metadata.climber_height}cm`);
    }

    const detailsText = details.length > 0 ? `\n${details.join(' | ')}` : '';

    // Show feedback with action button
    window.showFeedback(
      message + detailsText,
      'info',
      10000, // 10 seconds
      [
        {
          label: 'View Job',
          callback: () => {
            // Navigate to main page and select the job
            if (window.location.pathname !== '/') {
              window.location.href = `/?job=${job.id}`;
            } else {
              // If already on main page, trigger job selection
              const event = new CustomEvent('selectJob', { detail: { jobId: job.id } });
              window.dispatchEvent(event);
            }
          },
          style: 'primary'
        }
      ]
    );
  }

  /**
   * Initialize job monitoring
   */
  function initJobMonitoring() {
    // Initialize with current jobs to avoid showing notifications on page load
    fetch('/api/jobs')
      .then(res => {
        if (!res.ok) throw new Error('Failed to fetch jobs');
        return res.json();
      })
      .then(jobs => {
        // Mark all existing jobs as known
        knownJobIds = new Set(jobs.map(job => job.id));

        // Start polling for new jobs
        setInterval(checkForNewJobs, POLL_INTERVAL);

        console.log('[Job Notifications] Monitoring initialized with', knownJobIds.size, 'existing jobs');
      })
      .catch(error => {
        console.error('[Job Notifications] Failed to initialize:', error);
        // Still start polling even if initial fetch fails
        setInterval(checkForNewJobs, POLL_INTERVAL);
      });
  }

  // Initialize when page loads
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initJobMonitoring);
  } else {
    initJobMonitoring();
  }

  // Export for testing/debugging
  window.jobNotifications = {
    checkForNewJobs,
    getKnownJobIds: () => Array.from(knownJobIds)
  };
})();
