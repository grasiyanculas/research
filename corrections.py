"""Map (predicted wrong body part, current angles, reference profile) -> spoken phrase."""
from __future__ import annotations
import json
from typing import Dict, List, Optional
import numpy as np

import config as C


def load_reference(pose_name: str) -> dict:
    path = C.MODELS_DIR / f"reference_{pose_name}.json"
    return json.loads(path.read_text())


def _z(value: float, ref: dict) -> float:
    return (value - ref["mean"]) / max(ref["std"], 1.0)


def _side_visible(angle_name: str, vis: np.ndarray) -> bool:
    """Check that the joints involved in this angle are visible enough."""
    landmark_idx = {
        "left_elbow":   [C.LEFT_SHOULDER, C.LEFT_ELBOW, C.LEFT_WRIST],
        "right_elbow":  [C.RIGHT_SHOULDER, C.RIGHT_ELBOW, C.RIGHT_WRIST],
        "left_shoulder":[C.LEFT_ELBOW, C.LEFT_SHOULDER, C.LEFT_HIP],
        "right_shoulder":[C.RIGHT_ELBOW, C.RIGHT_SHOULDER, C.RIGHT_HIP],
        "left_hip":     [C.LEFT_SHOULDER, C.LEFT_HIP, C.LEFT_KNEE],
        "right_hip":    [C.RIGHT_SHOULDER, C.RIGHT_HIP, C.RIGHT_KNEE],
        "left_knee":    [C.LEFT_HIP, C.LEFT_KNEE, C.LEFT_ANKLE],
        "right_knee":   [C.RIGHT_HIP, C.RIGHT_KNEE, C.RIGHT_ANKLE],
        "left_ankle":   [C.LEFT_KNEE, C.LEFT_ANKLE, C.LEFT_FOOT_INDEX],
        "right_ankle":  [C.RIGHT_KNEE, C.RIGHT_ANKLE, C.RIGHT_FOOT_INDEX],
        "neck":         [C.NOSE, C.LEFT_SHOULDER, C.RIGHT_SHOULDER],
        "spine_tilt":   [C.LEFT_SHOULDER, C.RIGHT_SHOULDER, C.LEFT_HIP, C.RIGHT_HIP],
    }
    idxs = landmark_idx.get(angle_name, [])
    return all(vis[i] > 0 for i in idxs)


def choose_correction(
    pose_name: str,
    bodypart_probs: Dict[str, float],
    current_angles: Dict[str, float],
    reference: Dict[str, dict],
    visibility: np.ndarray,
) -> Optional[str]:
    """Return a phrase, or None if no clear deviation found.

    bodypart_probs: {"legs": 0.7, "neck": 0.2, ...} — predicted by bodypart head.
    current_angles: {"left_knee": 172.3, ...} degrees.
    reference:      {"left_knee": {"mean": ..., "std": ...}, ...}
    visibility:     33-d mask (1.0 visible, 0.0 not)."""
    # Rank candidate body parts by probability
    parts_ranked = sorted(bodypart_probs.items(), key=lambda kv: -kv[1])

    best = None  # (abs_z, angle, sign)
    for part_name, prob in parts_ranked:
        if prob < 0.4:
            break
        angle_candidates = C.BODY_PART_TO_ANGLES.get(part_name, [])
        for ang in angle_candidates:
            if ang not in reference or ang not in current_angles:
                continue
            if not _side_visible(ang, visibility):
                continue
            z = _z(current_angles[ang], reference[ang])
            if abs(z) < C.Z_ANOMALY:
                continue
            sign = -1 if z < 0 else +1
            if best is None or abs(z) > best[0]:
                best = (abs(z), ang, sign)
        if best is not None:
            break  # stick with the top-ranked part if we got a hit

    # Fallback: pick the joint with the largest |z| across ALL angles
    if best is None:
        for ang, ref in reference.items():
            if ang not in current_angles or not _side_visible(ang, visibility):
                continue
            z = _z(current_angles[ang], ref)
            if abs(z) < C.Z_ANOMALY:
                continue
            sign = -1 if z < 0 else +1
            if best is None or abs(z) > best[0]:
                best = (abs(z), ang, sign)

    if best is None:
        return None
    _, ang, sign = best
    phrase = C.ANGLE_DEVIATION_PHRASES.get((ang, sign))
    return phrase
