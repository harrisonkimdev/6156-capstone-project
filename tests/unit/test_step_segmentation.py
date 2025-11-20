from pose_ai.segmentation.steps import segment_steps_by_contacts


def test_segment_steps_creates_new_step_on_single_contact_change():
    rows = []
    for idx in range(10):
        rows.append(
            {
                "timestamp": idx / 25.0,
                "left_hand_contact_on": 1,
                "left_hand_contact_type": "hold",
                "left_hand_contact_hold": "H1",
                "right_hand_contact_on": 1,
                "right_hand_contact_type": "hold",
                "right_hand_contact_hold": "H2",
                "left_foot_contact_on": 1,
                "left_foot_contact_type": "hold",
                "left_foot_contact_hold": "F1",
                "right_foot_contact_on": 1,
                "right_foot_contact_type": "hold",
                "right_foot_contact_hold": "F2",
            }
        )
    # change right hand on later frames
    rows[5]["right_hand_contact_hold"] = "H3"
    rows[5]["right_hand_contact_type"] = "hold"
    segments = segment_steps_by_contacts(rows, fps=25.0, min_duration=0.1)
    assert len(segments) >= 2
    assert rows[0]["step_id"] == segments[0].step_id
    assert rows[-1]["step_id"] == segments[-1].step_id
