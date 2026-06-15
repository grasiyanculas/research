"""Train the pose classifier, per-pose correctness heads, and per-pose body-part heads."""
from __future__ import annotations
import json
import time
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

import config as C
from dataset import load_parquet, build_matrices, stratified_split
from features import flip_left_right


def augment(X: np.ndarray, y_pose: np.ndarray, y_correct: np.ndarray,
            y_bp: np.ndarray, seed: int = 0):
    """Horizontal flip augmentation only. Keeps subtle correctness boundaries intact."""
    flipped = np.stack([flip_left_right(x) for x in X])
    X_aug = np.concatenate([X, flipped], axis=0)
    y_pose_aug = np.concatenate([y_pose, y_pose])
    y_correct_aug = np.concatenate([y_correct, y_correct])
    y_bp_aug = np.concatenate([y_bp, y_bp], axis=0)
    return X_aug, y_pose_aug, y_correct_aug, y_bp_aug


def main():
    df = load_parquet().reset_index(drop=True)
    X, y_pose, y_correct, y_bp = build_matrices(df)
    print(f"[train] dataset: {X.shape}, poses={len(C.POSES)}, parts={len(C.BODY_PARTS)}")

    train_idx, val_idx = stratified_split(df)
    Xtr, Xva = X[train_idx], X[val_idx]
    yp_tr, yp_va = y_pose[train_idx], y_pose[val_idx]
    yc_tr, yc_va = y_correct[train_idx], y_correct[val_idx]
    ybp_tr, ybp_va = y_bp[train_idx], y_bp[val_idx]

    Xtr_a, yp_tr_a, yc_tr_a, ybp_tr_a = augment(Xtr, yp_tr, yc_tr, ybp_tr)
    print(f"[train] after augmentation: {Xtr_a.shape}")

    # ---- 1) Pose classifier ----------------------------------------------
    t0 = time.time()
    pose_clf = RandomForestClassifier(
        n_estimators=300, n_jobs=-1, class_weight="balanced", random_state=42,
    )
    pose_clf.fit(Xtr_a, yp_tr_a)
    pred = pose_clf.predict(Xva)
    pose_acc = accuracy_score(yp_va, pred)
    print(f"[train] pose classifier acc={pose_acc:.4f}  ({time.time()-t0:.1f}s)")
    joblib.dump(pose_clf, C.MODELS_DIR / "pose_clf.joblib")

    metrics = {"pose_acc": float(pose_acc), "per_pose": {}}

    # ---- 2) Per-pose correctness heads -----------------------------------
    for pose_id, pose_name in enumerate(C.POSES):
        mask_tr = yp_tr_a == pose_id
        mask_va = yp_va == pose_id
        if mask_tr.sum() < 10 or mask_va.sum() < 2:
            print(f"[train] {pose_name}: not enough data for correctness head, skipping")
            continue
        # Need both classes
        if len(np.unique(yc_tr_a[mask_tr])) < 2:
            print(f"[train] {pose_name}: only one correctness class present, skipping")
            continue
        clf = RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1, random_state=42,
        )
        clf.fit(Xtr_a[mask_tr], yc_tr_a[mask_tr])
        acc = accuracy_score(yc_va[mask_va], clf.predict(Xva[mask_va]))
        joblib.dump(clf, C.MODELS_DIR / f"correctness_{pose_name}.joblib")
        metrics["per_pose"].setdefault(pose_name, {})["correctness_acc"] = float(acc)
        flag = " (LOW)" if acc < 0.75 else ""
        print(f"[train] correctness  {pose_name:24s}  n_tr={mask_tr.sum():5d}  acc={acc:.3f}{flag}")

    # ---- 3) Per-pose body-part heads (multi-label) -----------------------
    for pose_id, pose_name in enumerate(C.POSES):
        # Train on wrong-only samples for this pose
        mask_tr = (yp_tr_a == pose_id) & (yc_tr_a == 0)
        mask_va = (yp_va == pose_id) & (yc_va == 0)
        if mask_tr.sum() < 20 or mask_va.sum() < 2:
            print(f"[train] {pose_name}: not enough wrong samples for body-part head, skipping")
            continue

        Y_tr = ybp_tr_a[mask_tr]
        # Keep only body parts that have at least one positive in training
        col_mask = Y_tr.sum(axis=0) > 0
        active_parts = [C.BODY_PARTS[i] for i in range(len(C.BODY_PARTS)) if col_mask[i]]
        if not active_parts:
            print(f"[train] {pose_name}: no active body parts, skipping")
            continue
        Y_tr_active = Y_tr[:, col_mask]
        Y_va_active = ybp_va[mask_va][:, col_mask]

        # MultiOutputClassifier needs each output to have both classes; filter further
        keep_cols = []
        for j in range(Y_tr_active.shape[1]):
            if len(np.unique(Y_tr_active[:, j])) == 2:
                keep_cols.append(j)
        if not keep_cols:
            print(f"[train] {pose_name}: every body-part column is constant, skipping")
            continue
        active_parts = [active_parts[j] for j in keep_cols]
        Y_tr_active = Y_tr_active[:, keep_cols]
        Y_va_active = Y_va_active[:, keep_cols]

        clf = MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=200, max_depth=None, min_samples_leaf=2,
                class_weight="balanced", n_jobs=-1, random_state=42,
            )
        )
        clf.fit(Xtr_a[mask_tr], Y_tr_active)
        pred = clf.predict(Xva[mask_va])
        f1 = f1_score(Y_va_active, pred, average="macro", zero_division=0)
        bundle = {"model": clf, "active_parts": active_parts}
        joblib.dump(bundle, C.MODELS_DIR / f"bodypart_{pose_name}.joblib")
        metrics["per_pose"].setdefault(pose_name, {})["bodypart_f1"] = float(f1)
        print(f"[train] bodypart     {pose_name:24s}  n_tr={mask_tr.sum():5d}  "
              f"parts={active_parts}  f1={f1:.3f}")

    (C.MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[train] wrote {C.MODELS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
