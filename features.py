"""Feature builder: normalize landmarks, compute joint angles, build 144-d vector."""
from __future__ import annotations
import numpy as np
from typing import Iterable

import config as C


def _to_array(landmarks) -> np.ndarray:
    """Accept MediaPipe NormalizedLandmarkList OR a raw (33,4) array. Return (33,4) float32."""
    if isinstance(landmarks, np.ndarray):
        return landmarks.astype(np.float32, copy=False)
    pts = np.empty((C.N_LANDMARKS, 4), dtype=np.float32)
    for i, lm in enumerate(landmarks.landmark):
        pts[i] = (lm.x, lm.y, lm.z, lm.visibility)
    return pts


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at vertex b formed by a-b-c, in degrees."""
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)
    if nba < 1e-6 or nbc < 1e-6:
        return 180.0
    cos = float(np.dot(ba, bc) / (nba * nbc))
    cos = max(-1.0, min(1.0, cos))
    return float(np.degrees(np.arccos(cos)))


def compute_angles(pts: np.ndarray) -> np.ndarray:
    """Return (12,) array of angle values in degrees. pts is (33, >=3)."""
    p = pts[:, :3]
    shoulder_mid = (p[C.LEFT_SHOULDER] + p[C.RIGHT_SHOULDER]) * 0.5
    hip_mid = (p[C.LEFT_HIP] + p[C.RIGHT_HIP]) * 0.5

    angles = np.zeros(C.N_ANGLES, dtype=np.float32)
    angles[0]  = _angle_deg(p[C.LEFT_SHOULDER],  p[C.LEFT_ELBOW],   p[C.LEFT_WRIST])
    angles[1]  = _angle_deg(p[C.RIGHT_SHOULDER], p[C.RIGHT_ELBOW],  p[C.RIGHT_WRIST])
    angles[2]  = _angle_deg(p[C.LEFT_ELBOW],     p[C.LEFT_SHOULDER],p[C.LEFT_HIP])
    angles[3]  = _angle_deg(p[C.RIGHT_ELBOW],    p[C.RIGHT_SHOULDER],p[C.RIGHT_HIP])
    angles[4]  = _angle_deg(p[C.LEFT_SHOULDER],  p[C.LEFT_HIP],     p[C.LEFT_KNEE])
    angles[5]  = _angle_deg(p[C.RIGHT_SHOULDER], p[C.RIGHT_HIP],    p[C.RIGHT_KNEE])
    angles[6]  = _angle_deg(p[C.LEFT_HIP],       p[C.LEFT_KNEE],    p[C.LEFT_ANKLE])
    angles[7]  = _angle_deg(p[C.RIGHT_HIP],      p[C.RIGHT_KNEE],   p[C.RIGHT_ANKLE])
    angles[8]  = _angle_deg(p[C.NOSE], shoulder_mid, hip_mid)   # neck

    # Spine tilt vs world y-axis (image coords: y grows downward, so use shoulder->hip vector).
    spine_vec = hip_mid - shoulder_mid
    vert = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    n = np.linalg.norm(spine_vec)
    if n < 1e-6:
        angles[9] = 0.0
    else:
        cos = float(np.dot(spine_vec, vert) / n)
        cos = max(-1.0, min(1.0, cos))
        angles[9] = float(np.degrees(np.arccos(cos)))

    angles[10] = _angle_deg(p[C.LEFT_KNEE],  p[C.LEFT_ANKLE],  p[C.LEFT_FOOT_INDEX])
    angles[11] = _angle_deg(p[C.RIGHT_KNEE], p[C.RIGHT_ANKLE], p[C.RIGHT_FOOT_INDEX])
    return angles


def normalize_landmarks(pts: np.ndarray) -> np.ndarray:
    """Translate to hip midpoint, scale by torso length. Return (33,3)."""
    p = pts[:, :3].copy()
    shoulder_mid = (p[C.LEFT_SHOULDER] + p[C.RIGHT_SHOULDER]) * 0.5
    hip_mid = (p[C.LEFT_HIP] + p[C.RIGHT_HIP]) * 0.5
    torso = float(np.linalg.norm(shoulder_mid - hip_mid))
    if torso < 1e-4:
        torso = 1.0
    p -= hip_mid
    p /= torso
    return p


def build_feature_vector(landmarks) -> np.ndarray:
    """Return (144,) feature vector. Accepts MediaPipe landmarks or (33,4) array."""
    pts = _to_array(landmarks)
    norm = normalize_landmarks(pts).reshape(-1)               # 99
    angles = compute_angles(pts) / 180.0                       # 12
    vis = (pts[:, 3] > C.VISIBILITY_OK).astype(np.float32)     # 33
    return np.concatenate([norm, angles, vis], dtype=np.float32)


def angles_from_features(feat: np.ndarray) -> dict:
    """Extract the 12 angle values (in degrees) from a 144-d feature vector."""
    a = feat[C.COORD_DIMS:C.COORD_DIMS + C.N_ANGLES] * 180.0
    return {name: float(a[i]) for i, name in enumerate(C.ANGLE_NAMES)}


def visibility_from_features(feat: np.ndarray) -> np.ndarray:
    """Extract the 33-d visibility mask from a 144-d feature vector."""
    return feat[C.COORD_DIMS + C.N_ANGLES:]


_LR_LANDMARK_SWAP = {
    1: 4, 2: 5, 3: 6, 7: 8,
    9: 10, 11: 12, 13: 14, 15: 16,
    17: 18, 19: 20, 21: 22, 23: 24,
    25: 26, 27: 28, 29: 30, 31: 32,
}
_LR_ANGLE_SWAP = {
    "left_elbow": "right_elbow", "left_shoulder": "right_shoulder",
    "left_hip": "right_hip", "left_knee": "right_knee", "left_ankle": "right_ankle",
}


def flip_left_right(feat: np.ndarray) -> np.ndarray:
    """Mirror a feature vector across the body's sagittal plane for augmentation.
    Swaps left/right landmarks, negates the normalized x coordinate, swaps left/right angles,
    swaps left/right visibility flags."""
    out = feat.copy()
    # 1. normalized coords (99) — negate x AND swap L/R landmark indices
    coords = out[:C.COORD_DIMS].reshape(C.N_LANDMARKS, 3).copy()
    coords[:, 0] *= -1.0
    swapped = coords.copy()
    for a, b in _LR_LANDMARK_SWAP.items():
        swapped[a] = coords[b]
        swapped[b] = coords[a]
    out[:C.COORD_DIMS] = swapped.reshape(-1)

    # 2. angles — swap L/R pairs
    angles = out[C.COORD_DIMS:C.COORD_DIMS + C.N_ANGLES].copy()
    name_to_idx = {n: i for i, n in enumerate(C.ANGLE_NAMES)}
    new_angles = angles.copy()
    for la, ra in _LR_ANGLE_SWAP.items():
        new_angles[name_to_idx[la]] = angles[name_to_idx[ra]]
        new_angles[name_to_idx[ra]] = angles[name_to_idx[la]]
    out[C.COORD_DIMS:C.COORD_DIMS + C.N_ANGLES] = new_angles

    # 3. visibility — swap L/R landmark indices
    vis = out[C.COORD_DIMS + C.N_ANGLES:].copy()
    new_vis = vis.copy()
    for a, b in _LR_LANDMARK_SWAP.items():
        new_vis[a] = vis[b]
        new_vis[b] = vis[a]
    out[C.COORD_DIMS + C.N_ANGLES:] = new_vis
    return out
