from pose_ai.recommendation.efficiency import StepEfficiencyComputer, score_steps
from pose_ai.segmentation.steps import StepSegment


def _make_row(timestamp: float, com_x: float, com_y: float, hold_suffix: str):
    return {
        "timestamp": timestamp,
        "body_scale": 0.4,
        "com_x": com_x,
        "com_y": com_y,
        "left_hand_contact_on": 1,
        "left_hand_contact_type": "hold",
        "left_hand_contact_hold": f"H{hold_suffix}",
        "left_hand_target_x": com_x - 0.1,
        "left_hand_target_y": com_y - 0.1,
        "right_hand_contact_on": 1,
        "right_hand_contact_type": "hold",
        "right_hand_contact_hold": f"H{hold_suffix}",
        "right_hand_target_x": com_x + 0.1,
        "right_hand_target_y": com_y - 0.1,
        "left_foot_contact_on": 1,
        "left_foot_contact_type": "hold",
        "left_foot_contact_hold": f"F{hold_suffix}",
        "left_foot_target_x": com_x - 0.1,
        "left_foot_target_y": com_y + 0.1,
        "right_foot_contact_on": 1,
        "right_foot_contact_type": "hold",
        "right_foot_contact_hold": f"F{hold_suffix}",
        "right_foot_target_x": com_x + 0.1,
        "right_foot_target_y": com_y + 0.1,
        "hip_center_x": com_x,
        "hip_center_y": com_y,
        "com_perp_wall": 0.0,
        "com_jerk": 0.01,
    }


def test_step_efficiency_scores_frames():
    frames = [_make_row(idx / 25.0, 0.5 + idx * 0.01, 0.4, "A") for idx in range(5)]
    segment = StepSegment(step_id=0, start_index=0, end_index=4, start_time=0.0, end_time=0.2)
    result = StepEfficiencyComputer().score_step(frames, segment)
    assert 0.0 <= result.score <= 1.0
    assert "stability" in result.components


def test_score_steps_batch():
    frames = [_make_row(idx / 25.0, 0.4, 0.4, "B") for idx in range(4)]
    segments = [StepSegment(step_id=0, start_index=0, end_index=3, start_time=0.0, end_time=0.12)]
    results = score_steps(frames, segments)
    assert len(results) == 1
    assert results[0].components["path_efficiency"] >= 0.0
