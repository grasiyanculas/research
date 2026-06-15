"""Load the parquet of extracted landmarks, build feature matrices and labels."""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple

import config as C
from features import build_feature_vector


def load_parquet() -> pd.DataFrame:
    if not C.KEYPOINTS_PARQUET.exists():
        raise FileNotFoundError(
            f"{C.KEYPOINTS_PARQUET} not found — run extract_keypoints.py first.")
    return pd.read_parquet(C.KEYPOINTS_PARQUET)


def build_matrices(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return X (N,144), y_pose (N,), y_correct (N,), y_bodyparts (N, len(BODY_PARTS))."""
    n = len(df)
    X = np.zeros((n, C.N_FEATURES), dtype=np.float32)
    y_pose = np.zeros(n, dtype=np.int64)
    y_correct = np.zeros(n, dtype=np.int64)
    n_parts = len(C.BODY_PARTS)
    y_bp = np.zeros((n, n_parts), dtype=np.float32)
    part_idx = {p: i for i, p in enumerate(C.BODY_PARTS)}

    for i, row in enumerate(df.itertuples(index=False)):
        # parquet returns an ndarray-of-ndarrays; stack into a proper (33,4) float array
        lm = np.stack([np.asarray(r, dtype=np.float32) for r in row.landmarks])
        X[i] = build_feature_vector(lm)
        y_pose[i] = row.pose_id
        y_correct[i] = row.is_right
        if not row.is_right:
            for p in row.body_parts:
                if p in part_idx:
                    y_bp[i, part_idx[p]] = 1.0
    return X, y_pose, y_correct, y_bp


def stratified_split(df: pd.DataFrame, val_frac: float = 0.2, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx) stratified by (pose_id, is_right)."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for _, group in df.groupby(["pose_id", "is_right"]):
        idx = group.index.to_numpy()
        rng.shuffle(idx)
        cut = int(len(idx) * (1.0 - val_frac))
        train_idx.append(idx[:cut])
        val_idx.append(idx[cut:])
    return np.concatenate(train_idx), np.concatenate(val_idx)
