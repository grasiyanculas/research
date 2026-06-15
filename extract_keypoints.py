"""Walk dataset, extract MediaPipe Pose keypoints from images and sampled video frames.

Outputs cache/keypoints.parquet with columns:
  pose_id, pose_name, source, is_right, step_folder, body_parts, landmarks, file_path, frame_idx
"""
from __future__ import annotations
import argparse
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from pathlib import Path
from typing import List, Tuple, Optional

# Import mediapipe BEFORE cv2 — on Windows + py3.12, importing cv2 first can
# break mediapipe's DLL search path. Also pre-import in main thread so DLLs
# initialize before any worker thread tries to use them.
import mediapipe as mp

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import config as C


_BODY_PART_SUFFIX_RE = re.compile(r"-\s*(.+)$")


def parse_body_parts(step_folder_name: str) -> List[str]:
    """Extract body-part tokens from a wrong-step folder name.
    E.g. 'Tadasana Wrong Step 2- Legs and Hand' -> ['legs', 'hands']."""
    m = _BODY_PART_SUFFIX_RE.search(step_folder_name)
    if not m:
        return []
    suffix = m.group(1).strip().lower()
    tokens = [t.strip() for t in suffix.split(" and ")]
    out = []
    for tok in tokens:
        canon = C.BODY_PART_ALIASES.get(tok)
        if canon is None:
            # try stripping parenthetical: 'back (overbend)' -> 'back'
            base = tok.split("(")[0].strip()
            canon = C.BODY_PART_ALIASES.get(base)
        if canon and canon not in out:
            out.append(canon)
    return out


def discover_image_samples() -> List[dict]:
    """Walk Images/ and return one dict per JPG."""
    samples = []
    for pose in C.POSES:
        pose_dir = C.IMAGES_DIR / pose
        if not pose_dir.exists():
            continue
        for side_dir in pose_dir.iterdir():
            if not side_dir.is_dir():
                continue
            is_right = "right" in side_dir.name.lower()
            for step_dir in side_dir.iterdir():
                if not step_dir.is_dir():
                    continue
                body_parts = [] if is_right else parse_body_parts(step_dir.name)
                for img in step_dir.glob("*.jpg"):
                    samples.append({
                        "pose_id": C.POSE_TO_ID[pose],
                        "pose_name": pose,
                        "source": "image",
                        "is_right": int(is_right),
                        "step_folder": step_dir.name,
                        "body_parts": body_parts,
                        "file_path": str(img),
                        "frame_idx": -1,
                    })
    return samples


def discover_video_jobs() -> List[dict]:
    """One dict per video file; frame sampling happens inside the worker."""
    jobs = []
    for pose in C.POSES:
        pose_dir = C.VIDEOS_DIR / pose
        if not pose_dir.exists():
            continue
        for side_dir in pose_dir.iterdir():
            if not side_dir.is_dir():
                continue
            is_right = "right" in side_dir.name.lower()
            # Videos sit directly under the side directory; not in step subfolders.
            for vid in side_dir.glob("*.mp4"):
                body_parts = [] if is_right else parse_body_parts(side_dir.name)
                jobs.append({
                    "pose_id": C.POSE_TO_ID[pose],
                    "pose_name": pose,
                    "source": "video",
                    "is_right": int(is_right),
                    "step_folder": side_dir.name,
                    "body_parts": body_parts,
                    "file_path": str(vid),
                })
    return jobs


# ---------------------------------------------------------------------------
# Worker: each thread owns its own MediaPipe Pose instance via thread-local.
# Multiprocessing was tried first but mediapipe 0.10.18 fails to load its DLLs
# in spawned children on Windows + Python 3.12. MediaPipe inference is C++ and
# releases the GIL, so threads still parallelize the heavy work.
# ---------------------------------------------------------------------------
_TLS = threading.local()


def _get_pose():
    p = getattr(_TLS, "pose", None)
    if p is None:
        p = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
        )
        _TLS.pose = p
    return p


def _extract_one_image(path: str) -> Optional[np.ndarray]:
    """Return (33,4) landmarks array or None."""
    img = cv2.imread(path)
    if img is None:
        return None
    return _process_bgr(img)


def _process_bgr(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    pose = _get_pose()
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)
    if res.pose_landmarks is None:
        return None
    arr = np.empty((C.N_LANDMARKS, 4), dtype=np.float32)
    for i, lm in enumerate(res.pose_landmarks.landmark):
        arr[i] = (lm.x, lm.y, lm.z, lm.visibility)
    if float(arr[:, 3].mean()) < C.MIN_MEAN_VISIBILITY:
        return None
    return arr


def worker_image(sample: dict) -> Optional[dict]:
    lm = _extract_one_image(sample["file_path"])
    if lm is None:
        return None
    return {**sample, "landmarks": lm.tolist()}


def worker_video(job: dict) -> List[dict]:
    cap = cv2.VideoCapture(job["file_path"])
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(fps / C.VIDEO_SAMPLE_FPS)))
    out = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            lm = _process_bgr(frame)
            if lm is not None:
                out.append({**job, "landmarks": lm.tolist(), "frame_idx": frame_idx})
        frame_idx += 1
    cap.release()
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, limit total samples (for smoke test).")
    ap.add_argument("--only-pose", type=str, default="",
                    help="Restrict to a single pose name (for smoke test).")
    ap.add_argument("--images-only", action="store_true",
                    help="Skip video frame sampling.")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    args = ap.parse_args()

    print(f"[extract] discovering samples...")
    img_samples = discover_image_samples()
    vid_jobs = [] if args.images_only else discover_video_jobs()

    if args.only_pose:
        img_samples = [s for s in img_samples if s["pose_name"] == args.only_pose]
        vid_jobs = [j for j in vid_jobs if j["pose_name"] == args.only_pose]
    if args.limit:
        img_samples = img_samples[: args.limit]
        vid_jobs = vid_jobs[: max(1, args.limit // 100)]

    print(f"[extract] {len(img_samples)} image samples, {len(vid_jobs)} video jobs, "
          f"workers={args.workers}")

    rows: List[dict] = []
    t0 = time.time()

    # Images
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(worker_image, s): s for s in img_samples}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="images"):
            r = fut.result()
            if r is not None:
                rows.append(r)

    # Videos
    if vid_jobs:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(worker_video, j): j for j in vid_jobs}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="videos"):
                rows.extend(fut.result())

    print(f"[extract] kept {len(rows)} rows in {time.time()-t0:.1f}s")
    if not rows:
        print("[extract] no rows extracted — aborting", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)
    # body_parts is a list — pyarrow handles that natively.
    df.to_parquet(C.KEYPOINTS_PARQUET, index=False)
    print(f"[extract] wrote {C.KEYPOINTS_PARQUET}")
    print(df.groupby(["pose_name", "is_right"]).size())


if __name__ == "__main__":
    main()
