#!/usr/bin/env python3
"""Apply contact detection to already extracted frames."""

import sys
from pathlib import Path
import json

sys.path.insert(0, 'src')

from pose_ai.pose.estimator import PoseEstimator
from pose_ai.data.advanced_sampler import compute_limb_velocity, detect_contact_moments

# Use already extracted frames
frame_dir = Path('data/test_frames/IMG_3571')
frame_files = sorted(frame_dir.glob('IMG_3571_frame_*.jpg'))

print(f'Found {len(frame_files)} frames')
print('Running pose estimation...')

# Run pose estimation
estimator = PoseEstimator()
pose_frames = estimator.process_paths(frame_files)
estimator.close()

print(f'Pose estimation complete: {len(pose_frames)} poses detected')

# Compute limb velocities
print('Computing limb velocities...')
velocities = compute_limb_velocity(pose_frames)

# Detect contact moments
print('Detecting contact moments...')
contact_indices = detect_contact_moments(
    velocities,
    velocity_spike_threshold=0.05,
    velocity_low_threshold=0.01,
    hold_duration_frames=3,
)

print(f'\nDetected {len(contact_indices)} contact moments:')
print(f'Contact frame indices: {contact_indices[:20]}...' if len(contact_indices) > 20 else f'Contact frame indices: {contact_indices}')

# Save result
result = {
    'total_frames': len(frame_files),
    'contact_moments': len(contact_indices),
    'contact_frame_indices': contact_indices,
    'selected_frames': [str(frame_files[i]) for i in contact_indices if i < len(frame_files)]
}

output_file = frame_dir / 'contact_detection_result.json'
output_file.write_text(json.dumps(result, indent=2))
print(f'\nResult saved to: {output_file}')
