#!/usr/bin/env python3
"""Quick test script for frame extraction."""

import sys
from pathlib import Path

sys.path.insert(0, 'src')

from pose_ai.data.advanced_sampler import extract_frames_with_motion

video_path = Path('data/test_video/IMG_3571.mov')
output_root = Path('data/workflow_frames/test_direct_test')

print(f'Video exists: {video_path.exists()}')
print(f'Starting extraction...')

try:
    result = extract_frames_with_motion(
        video_path,
        output_root=output_root,
        motion_threshold=5.0,
        similarity_threshold=0.8,
        write_manifest=True,
        overwrite=True,
    )

    print(f'\nSaved frames: {result.saved_frames}')
    print(f'Frame directory: {result.frame_directory}')
except Exception as e:
    print(f'ERROR: {type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()
