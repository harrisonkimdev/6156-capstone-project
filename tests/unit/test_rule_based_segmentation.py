from __future__ import annotations

from pose_ai.segmentation import FrameMetrics, segment_by_activity


def test_segment_by_activity_splits_movement_and_rest():
    frames = []
    timestamp = 0.0
    for motion in [0.05] * 5 + [0.25] * 6 + [0.1] * 4:
        frames.append(FrameMetrics(timestamp=timestamp, motion_score=motion))
        timestamp += 0.5

    segments = segment_by_activity(frames, motion_threshold=0.18, min_segment_duration=1.0)
    assert [segment.label for segment in segments] == ["rest", "movement", "rest"]


def test_hold_change_bonus_promotes_segment_switch():
    frames = [
        FrameMetrics(timestamp=0.0, motion_score=0.1, hold_changed=False),
        FrameMetrics(timestamp=0.5, motion_score=0.12, hold_changed=True),
        FrameMetrics(timestamp=1.0, motion_score=0.11, hold_changed=True),
        FrameMetrics(timestamp=1.5, motion_score=0.08, hold_changed=False),
        FrameMetrics(timestamp=2.0, motion_score=0.05, hold_changed=False),
    ]

    segments = segment_by_activity(frames, motion_threshold=0.18, sustain_frames=1, min_segment_duration=0.5)
    assert len(segments) >= 2
    assert any(seg.label == "movement" for seg in segments)
