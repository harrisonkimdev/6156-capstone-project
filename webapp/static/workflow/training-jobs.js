/**
 * Training Jobs Module
 * 
 * Manages training job listing, status, and actions.
 */

/**
 * Load and display training jobs (table format)
 */
async function loadTrainingJobs() {
  try {
    const response = await fetch('/api/yolo/train/jobs');
    if (!response.ok) return;

    const data = await response.json();
    const container = document.getElementById('jobs-container');
    if (!container) return;

    if (data.jobs.length === 0) {
      container.innerHTML = '<p style="color: #888; text-align: center; padding: 20px;">No training jobs yet. Train after labeling/selection.</p>';
      return;
    }

    // Sort jobs by created_at (newest first)
    const sortedJobs = data.jobs.sort((a, b) => {
      const dateA = new Date(a.created_at || 0);
      const dateB = new Date(b.created_at || 0);
      return dateB - dateA;
    });

    container.innerHTML = `
      <div class="table-wrapper">
        <table class="jobs-table">
          <thead>
            <tr>
              <th>Job ID</th>
              <th>Status</th>
              <th>Model</th>
              <th>Created</th>
              <th>Completed</th>
              <th>Results</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            ${sortedJobs.map(job => {
      const createdDate = job.created_at ? new Date(job.created_at).toLocaleString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      }) : '—';

      const completedDate = job.completed_at ? new Date(job.completed_at).toLocaleString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      }) : (job.status === 'running' ? 'In progress...' : '—');

      // Format metrics
      let metricsText = '—';
      if (job.status === 'completed' && job.metrics) {
        if (job.metrics.mAP50) {
          metricsText = `mAP50: ${(job.metrics.mAP50 * 100).toFixed(1)}%`;
        } else if (job.metrics.map) {
          metricsText = `mAP: ${(job.metrics.map * 100).toFixed(1)}%`;
        }
      }

      // Actions column
      let actions = '';
      if (job.status === 'running') {
        actions = `<button class="btn-cancel-job" onclick="cancelTrainingJob('${job.id}')" style="padding: 4px 8px; font-size: 12px; background: var(--danger); color: white; border: none; border-radius: 4px; cursor: pointer;">Cancel</button>`;
      } else if (job.status === 'completed') {
        if (job.gcs_uri) {
          actions = `<span style="color: #00cc66; font-size: 11px;">✓ Uploaded</span>`;
        } else {
          actions = `<button onclick="uploadJobToGCS('${job.id}')" style="padding: 4px 8px; font-size: 12px; background: #0066cc; color: white; border: none; border-radius: 4px; cursor: pointer;">Upload</button>`;
        }
      } else if (job.status === 'failed') {
        actions = `<span style="color: #f85149; font-size: 11px;">Failed</span>`;
      }

      return `
                <tr>
                  <td style="font-family: monospace; font-size: 12px;">${job.id.substring(0, 8)}...</td>
                  <td><span class="badge badge-${job.status}">${job.status.toUpperCase()}</span></td>
                  <td style="font-size: 12px;">${job.model || 'YOLOv8'}</td>
                  <td style="font-size: 12px; color: #aaa;">${createdDate}</td>
                  <td style="font-size: 12px; color: #aaa;">${completedDate}</td>
                  <td style="font-size: 12px;">${metricsText}</td>
                  <td>${actions}</td>
                </tr>
              `;
    }).join('')}
          </tbody>
        </table>
      </div>
    `;

    // Enable GCS upload button if training completed
    const completedJobs = data.jobs.filter(j => j.status === 'completed');
    if (completedJobs.length > 0 && WorkflowState.getCurrentTrainingJobId()) {
      const currentJob = data.jobs.find(j => j.id === WorkflowState.getCurrentTrainingJobId());
      if (currentJob && currentJob.status === 'completed') {
        const btnUploadGCS = document.getElementById('btn-upload-gcs');
        if (btnUploadGCS) {
          btnUploadGCS.disabled = false;
        }
      }
    }

  } catch (error) {
    console.error('Failed to load training jobs:', error);
  }
}

/**
 * Upload job model to GCS
 * @param {string} jobId - Job ID
 */
async function uploadJobToGCS(jobId) {
  try {
    const response = await fetch(`/api/yolo/train/jobs/${jobId}/upload`, {
      method: 'POST',
    });

    if (!response.ok) {
      throw new Error(`Upload failed: ${response.statusText}`);
    }

    const data = await response.json();
    alert(`Model uploaded to: ${data.gcs_uri}`);
    loadTrainingJobs();

  } catch (error) {
    alert(`Error: ${error.message}`);
  }
}

/**
 * Cancel a training job
 * @param {string} jobId - Job ID
 */
async function cancelTrainingJob(jobId) {
  if (!confirm(`Are you sure you want to cancel training job ${jobId.substring(0, 8)}...?`)) {
    return;
  }

  try {
    // Try to call cancel endpoint (may not exist yet)
    const response = await fetch(`/api/yolo/train/jobs/${jobId}/cancel`, {
      method: 'POST',
    });

    if (!response.ok) {
      if (response.status === 404) {
        // API endpoint not implemented yet
        alert('Cancel functionality is not yet implemented on the server. The job will continue running.');
        return;
      }
      throw new Error(`Cancel failed: ${response.statusText}`);
    }

    const data = await response.json();
    alert(`Training job cancelled successfully.`);
    loadTrainingJobs();

  } catch (error) {
    console.error('Failed to cancel job:', error);
    alert(`Error: ${error.message}\n\nNote: Cancel functionality may not be fully implemented yet.`);
  }
}
