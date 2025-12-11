/**
 * Summary Module
 * 
 * Loads and displays dashboard summary information.
 */

/**
 * Load and display summary information
 */
async function loadSummaryInfo() {
  try {
    // Load labeled images count from hold detection pool
    const holdPoolResponse = await fetch('/api/workflow/pool/hold-detection');
    let labeledCount = 0;
    if (holdPoolResponse.ok) {
      const holdData = await holdPoolResponse.json();
      labeledCount = holdData.total_labeled_segments || 0;
    }

    // Load training jobs to get last trained time and model status
    const jobsResponse = await fetch('/api/yolo/train/jobs');
    let lastTrained = 'Never';
    let modelStatus = 'No model';

    if (jobsResponse.ok) {
      const jobsData = await jobsResponse.json();
      if (jobsData.jobs && jobsData.jobs.length > 0) {
        // Sort by created_at to get the most recent
        const sortedJobs = jobsData.jobs.sort((a, b) => {
          const dateA = new Date(a.created_at || 0);
          const dateB = new Date(b.created_at || 0);
          return dateB - dateA;
        });

        const latestJob = sortedJobs[0];

        // Format last trained date
        if (latestJob.completed_at) {
          const date = new Date(latestJob.completed_at);
          lastTrained = date.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
          });
        } else if (latestJob.created_at) {
          const date = new Date(latestJob.created_at);
          lastTrained = date.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
          });
        }

        // Get model status
        const completedJobs = sortedJobs.filter(j => j.status === 'completed');
        if (completedJobs.length > 0) {
          const bestJob = completedJobs[0];
          modelStatus = `${bestJob.model || 'YOLOv8'} / Ready`;
        } else if (latestJob.status === 'running') {
          modelStatus = `${latestJob.model || 'YOLOv8'} / Training...`;
        } else if (latestJob.status === 'pending') {
          modelStatus = `${latestJob.model || 'YOLOv8'} / Pending`;
        } else if (latestJob.status === 'failed') {
          modelStatus = `${latestJob.model || 'YOLOv8'} / Failed`;
        }
      }
    }

    // Update summary UI
    const labeledCountEl = document.getElementById('summary-labeled-count');
    const lastTrainedEl = document.getElementById('summary-last-trained');
    const modelStatusEl = document.getElementById('summary-model-status');

    if (labeledCountEl) {
      labeledCountEl.textContent = labeledCount > 0 ? `${labeledCount} Labeled Images` : 'No labeled images';
    }
    if (lastTrainedEl) {
      lastTrainedEl.textContent = lastTrained;
    }
    if (modelStatusEl) {
      modelStatusEl.textContent = modelStatus;
    }

  } catch (error) {
    console.error('Failed to load summary info:', error);
  }
}
