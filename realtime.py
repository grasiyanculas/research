"""Real-time yoga pose correction app.

Captures webcam, runs MediaPipe Pose, classifies pose + correctness, generates
correction phrase, speaks via TTS, draws overlay. Mirror is applied at display
time only — MediaPipe sees the raw frame so left/right corrections are anatomic.
"""
from __future__ import annotations
import json
import time
from collections import Counter, deque
from pathlib import Path

import mediapipe as mp   # must precede cv2 on Windows + py3.12 for DLL load
import cv2
import joblib
import numpy as np

import config as C
from features import build_feature_vector, angles_from_features, visibility_from_features
from corrections import choose_correction, load_reference
from tts import TTSWorker


def _load_models():
    pose_clf = joblib.load(C.MODELS_DIR / "pose_clf.joblib")
    correctness = {}
    bodypart = {}
    references = {}
    for pose in C.POSES:
        cp = C.MODELS_DIR / f"correctness_{pose}.joblib"
        bp = C.MODELS_DIR / f"bodypart_{pose}.joblib"
        rp = C.MODELS_DIR / f"reference_{pose}.json"
        if cp.exists():
            correctness[pose] = joblib.load(cp)
        if bp.exists():
            bodypart[pose] = joblib.load(bp)
        if rp.exists():
            references[pose] = load_reference(pose)
    return pose_clf, correctness, bodypart, references


def _bodypart_probabilities(bundle, feature: np.ndarray) -> dict:
    """Run the body-part multi-output classifier and return {part: P(part=1)}."""
    clf = bundle["model"]
    parts = bundle["active_parts"]
    out = {}
    # MultiOutputClassifier.predict_proba returns a list of (N,2) arrays
    probas = clf.predict_proba(feature.reshape(1, -1))
    for part, p in zip(parts, probas):
        out[part] = float(p[0, 1])
    return out


def _draw_overlay(frame, pose_name, correct, phrase, fps, status):
    h, w = frame.shape[:2]
    bar_h = 110
    cv2.rectangle(frame, (0, 0), (w, bar_h), (20, 20, 20), -1)
    if status:
        cv2.putText(frame, status, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (200, 200, 200), 2)
    else:
        col = (50, 200, 50) if correct else (50, 50, 230)
        label = "Correct" if correct else "Wrong"
        cv2.putText(frame, f"{pose_name}  -  {label}", (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
        if not correct and phrase:
            cv2.putText(frame, phrase, (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
    cv2.putText(frame, f"{fps:5.1f} FPS", (w - 130, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


def _draw_skeleton(frame, results, mp_drawing, mp_pose):
    if results.pose_landmarks is not None:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2),
        )


def main():
    print("[realtime] loading models...")
    pose_clf, correctness_models, bodypart_models, references = _load_models()
    print(f"[realtime] poses={len(C.POSES)}  correctness={len(correctness_models)}  "
          f"bodypart={len(bodypart_models)}")

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    tts = TTSWorker()
    tts.speak_async("Yoga pose correction system ready.")

    feat_buf: deque = deque(maxlen=C.SMOOTH_WINDOW)
    vote_buf: deque = deque(maxlen=C.VOTE_WINDOW)
    current_pose: str | None = None
    stable_since = 0.0
    last_spoken_at = 0.0
    last_phrase: str | None = None
    fps_t0 = time.time()
    fps_n = 0
    fps_val = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            status = ""
            display_pose = current_pose or ""
            display_correct = True
            display_phrase = ""

            if results.pose_landmarks is None:
                status = "No person detected"
                feat_buf.clear()
                vote_buf.clear()
                current_pose = None
            else:
                feat = build_feature_vector(results.pose_landmarks)
                feat_buf.append(feat)
                if len(feat_buf) < 5:
                    status = "Calibrating..."
                else:
                    smoothed = np.median(np.stack(feat_buf), axis=0).astype(np.float32)
                    pose_id = int(pose_clf.predict(smoothed.reshape(1, -1))[0])
                    voted_id = pose_id
                    vote_buf.append(pose_id)
                    if len(vote_buf) >= 5:
                        voted_id = Counter(vote_buf).most_common(1)[0][0]
                    voted_name = C.POSES[voted_id]

                    if voted_name != current_pose:
                        current_pose = voted_name
                        stable_since = time.time()

                    if time.time() - stable_since < C.POSE_STABLE_SECONDS:
                        status = f"Detecting... ({voted_name})"
                    else:
                        clf_c = correctness_models.get(voted_name)
                        if clf_c is None:
                            status = f"{voted_name}: no correctness model"
                        else:
                            p_correct = float(clf_c.predict_proba(smoothed.reshape(1, -1))[0, 1])
                            display_pose = voted_name
                            display_correct = p_correct >= C.CORRECT_THRESHOLD
                            if not display_correct:
                                bundle = bodypart_models.get(voted_name)
                                ref = references.get(voted_name, {})
                                cur_angles = angles_from_features(smoothed)
                                vis = visibility_from_features(smoothed)
                                bp_probs = (_bodypart_probabilities(bundle, smoothed)
                                            if bundle else {})
                                phrase = choose_correction(
                                    voted_name, bp_probs, cur_angles, ref, vis)
                                display_phrase = phrase or "adjust your posture"
                                now = time.time()
                                if (now - last_spoken_at >= C.TTS_COOLDOWN_SECONDS
                                        and display_phrase != last_phrase):
                                    tts.speak_async(display_phrase)
                                    last_spoken_at = now
                                    last_phrase = display_phrase
                            else:
                                last_phrase = None  # reset cooldown phrase when correct

            # ---- draw ----
            _draw_skeleton(frame, results, mp_drawing, mp_pose)
            display = cv2.flip(frame, 1)   # mirror for display only
            _draw_overlay(display, display_pose, display_correct, display_phrase, fps_val, status)
            cv2.imshow("Yoga Pose Correction (press Q to quit)", display)

            fps_n += 1
            if fps_n >= 15:
                fps_val = fps_n / (time.time() - fps_t0)
                fps_t0 = time.time()
                fps_n = 0

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tts.stop()


if __name__ == "__main__":
    main()
