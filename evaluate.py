"""Offline evaluation: replay a dataset video through the realtime pipeline and print outputs."""
from __future__ import annotations
import argparse
import time
from collections import Counter, deque

import mediapipe as mp   # must precede cv2 on Windows + py3.12 for DLL load
import cv2
import joblib
import numpy as np

import config as C
from features import build_feature_vector, angles_from_features, visibility_from_features
from corrections import choose_correction, load_reference
from realtime import _load_models, _bodypart_probabilities


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to .mp4 to replay")
    args = ap.parse_args()

    pose_clf, correctness_models, bodypart_models, references = _load_models()
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(model_complexity=1, smooth_landmarks=True,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.video}")

    feat_buf: deque = deque(maxlen=C.SMOOTH_WINDOW)
    votes = Counter()
    phrase_hist = Counter()
    n_frames = 0
    n_with_pose = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n_frames += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks is None:
            continue
        n_with_pose += 1
        feat = build_feature_vector(res.pose_landmarks)
        feat_buf.append(feat)
        if len(feat_buf) < 5:
            continue
        smoothed = np.median(np.stack(feat_buf), axis=0).astype(np.float32)
        pose_id = int(pose_clf.predict(smoothed.reshape(1, -1))[0])
        votes[C.POSES[pose_id]] += 1
        pose_name = C.POSES[pose_id]
        clf_c = correctness_models.get(pose_name)
        if clf_c is None:
            continue
        p_correct = float(clf_c.predict_proba(smoothed.reshape(1, -1))[0, 1])
        if p_correct < C.CORRECT_THRESHOLD:
            bundle = bodypart_models.get(pose_name)
            ref = references.get(pose_name, {})
            cur_angles = angles_from_features(smoothed)
            vis = visibility_from_features(smoothed)
            bp_probs = _bodypart_probabilities(bundle, smoothed) if bundle else {}
            phrase = choose_correction(pose_name, bp_probs, cur_angles, ref, vis)
            phrase_hist[phrase or "(none)"] += 1

    cap.release()
    print(f"[eval] frames={n_frames}  with_pose={n_with_pose}")
    print(f"[eval] pose votes: {votes.most_common()}")
    print(f"[eval] phrase distribution:")
    for ph, n in phrase_hist.most_common():
        print(f"   {n:5d}  {ph}")


if __name__ == "__main__":
    main()
