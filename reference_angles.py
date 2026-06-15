"""Compute per-pose reference angle profiles from Right-Steps samples only."""
from __future__ import annotations
import json
import numpy as np

import config as C
from dataset import load_parquet, build_matrices
from features import angles_from_features


def main():
    df = load_parquet()
    X, y_pose, y_correct, _ = build_matrices(df)

    for pose_id, pose_name in enumerate(C.POSES):
        mask = (y_pose == pose_id) & (y_correct == 1)
        if mask.sum() == 0:
            print(f"[ref] WARN no right-step samples for {pose_name}")
            continue
        feats = X[mask]
        # angles are stored normalized (/180) in features; recover degrees per sample
        angle_block = feats[:, C.COORD_DIMS:C.COORD_DIMS + C.N_ANGLES] * 180.0
        profile = {}
        for i, name in enumerate(C.ANGLE_NAMES):
            vals = angle_block[:, i]
            profile[name] = {"mean": float(np.mean(vals)), "std": float(max(np.std(vals), 1.0))}
        out = C.MODELS_DIR / f"reference_{pose_name}.json"
        out.write_text(json.dumps(profile, indent=2))
        print(f"[ref] {pose_name}: {mask.sum()} samples -> {out.name}")


if __name__ == "__main__":
    main()
