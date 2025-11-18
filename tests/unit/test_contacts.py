import math

from pose_ai.features.contacts import ContactParams, annotate_techniques, apply_contact_filter


def _base_row():
    return {
        "timestamp": 0.0,
        "body_scale": 0.4,
        "left_hand_distance": 0.05,
        "left_hand_vx": 0.0,
        "left_hand_vy": 0.0,
        "left_hand_target": "H1",
        "left_hand_z": 0.0,
        "left_foot_distance": 0.50,
        "left_foot_vx": 0.0,
        "left_foot_vy": 0.0,
        "left_foot_target": None,
        "left_foot_z": 0.0,
        "right_foot_distance": 0.50,
        "right_foot_vx": 0.0,
        "right_foot_vy": 0.0,
        "right_foot_target": None,
        "right_foot_z": 0.0,
        "hip_center_x": 0.4,
        "left_foot_x": 0.3,
        "right_foot_x": 0.5,
        "left_knee_x": 0.32,
        "right_knee_x": 0.48,
    }


def test_contact_filter_with_hysteresis():
    rows = []
    for idx in range(5):
        row = _base_row()
        row["timestamp"] = idx / 25.0
        rows.append(row)
    apply_contact_filter(rows, ContactParams(min_on_frames=2, r_on=0.2, r_off=0.3))
    assert rows[0]["left_hand_contact_on"] == 0
    assert rows[2]["left_hand_contact_on"] == 1
    assert rows[-1]["left_hand_contact_hold"] == "H1"


def test_smear_detection_and_techniques():
    rows = []
    for idx in range(4):
        row = _base_row()
        row["timestamp"] = idx / 25.0
        row["right_foot_distance"] = 0.6
        row["right_foot_vx"] = 0.0
        row["right_foot_vy"] = 0.0
        rows.append(row)
    apply_contact_filter(rows, ContactParams(min_on_frames=1))
    assert rows[-1]["right_foot_contact_type"] == "smear"
    annotate_techniques(rows)
    assert "technique_bicycle" in rows[-1]
