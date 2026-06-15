"""Configuration: dataset paths, pose list, landmark indices, body-part lookups, thresholds."""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).parent
DATASET_DIR = ROOT / "Yoga Postures Dataset"
IMAGES_DIR = DATASET_DIR / "Images"
VIDEOS_DIR = DATASET_DIR / "Videos"
CACHE_DIR = ROOT / "cache"
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"

for d in (CACHE_DIR, MODELS_DIR, LOGS_DIR):
    d.mkdir(exist_ok=True)

KEYPOINTS_PARQUET = CACHE_DIR / "keypoints.parquet"
FEATURES_NPZ = CACHE_DIR / "features.npz"

POSES = [
    "Anantasana",
    "Ardhakati Chakrasana",
    "Bhujangasana",
    "Kati Chakrasana",
    "Marjariasana",
    "Parvatasana",
    "Sarvangasana",
    "Tadasana",
    "Vajrasana",
    "Viparita Karani",
]
POSE_TO_ID = {p: i for i, p in enumerate(POSES)}

# MediaPipe Pose landmark indices (BlazePose 33-point topology)
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 31, 32

N_LANDMARKS = 33
COORD_DIMS = N_LANDMARKS * 3              # 99
N_ANGLES = 12
N_VISIBILITY = N_LANDMARKS                # 33
N_FEATURES = COORD_DIMS + N_ANGLES + N_VISIBILITY  # 144

ANGLE_NAMES = [
    "left_elbow", "right_elbow",
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "neck", "spine_tilt",
    "left_ankle", "right_ankle",
]

# Canonical body parts (lowercased keys). Multi-word folder suffixes split on " and ".
BODY_PARTS = [
    "legs", "hands", "knees", "neck", "head", "back", "ankle", "palms", "standing",
]
BODY_PART_ALIASES = {
    "leg": "legs", "legs": "legs",
    "hand": "hands", "hands": "hands",
    "knee": "knees", "knees": "knees",
    "neck": "neck",
    "head": "head",
    "back": "back",
    "back (overbend)": "back",
    "back (bending)": "back",
    "ankle": "ankle", "ankles": "ankle",
    "palms": "palms", "palm": "palms",
    "standing": "standing",
}

# Map body part -> candidate angle names to inspect for deviation.
BODY_PART_TO_ANGLES = {
    "legs":     ["left_knee", "right_knee", "left_hip", "right_hip"],
    "knees":    ["left_knee", "right_knee"],
    "hands":    ["left_elbow", "right_elbow", "left_shoulder", "right_shoulder"],
    "palms":    ["left_elbow", "right_elbow"],
    "neck":     ["neck"],
    "head":     ["neck"],
    "back":     ["spine_tilt"],
    "ankle":    ["left_ankle", "right_ankle"],
    "standing": ["spine_tilt", "left_hip", "right_hip"],
}

# Phrase templates by (angle, sign-of-deviation).
# sign = -1 means current < reference; sign = +1 means current > reference.
ANGLE_DEVIATION_PHRASES = {
    ("left_elbow",    -1): "extend your left arm",
    ("left_elbow",    +1): "bend your left elbow",
    ("right_elbow",   -1): "extend your right arm",
    ("right_elbow",   +1): "bend your right elbow",
    ("left_shoulder", -1): "lift your left arm higher",
    ("left_shoulder", +1): "lower your left arm",
    ("right_shoulder",-1): "lift your right arm higher",
    ("right_shoulder",+1): "lower your right arm",
    ("left_hip",      -1): "open your left hip more",
    ("left_hip",      +1): "lower your left leg",
    ("right_hip",     -1): "open your right hip more",
    ("right_hip",     +1): "lower your right leg",
    ("left_knee",     -1): "straighten your left knee",
    ("left_knee",     +1): "bend your left knee more",
    ("right_knee",    -1): "straighten your right knee",
    ("right_knee",    +1): "bend your right knee more",
    ("left_ankle",    -1): "press your left heel down",
    ("left_ankle",    +1): "point your left foot",
    ("right_ankle",   -1): "press your right heel down",
    ("right_ankle",   +1): "point your right foot",
    ("neck",          -1): "lift your head",
    ("neck",          +1): "tuck your chin slightly",
    ("spine_tilt",    -1): "stand straighter",
    ("spine_tilt",    +1): "keep your back upright",
}

# Inverted poses (MediaPipe is less reliable; relax thresholds).
INVERTED_POSES = {"Sarvangasana", "Viparita Karani"}

# ---- thresholds ----
VISIBILITY_OK = 0.5            # joint considered visible
MIN_MEAN_VISIBILITY = 0.3      # reject samples below this in offline extraction
CORRECT_THRESHOLD = 0.55       # P(correct) >= this -> green
Z_ANOMALY = 1.5                # |z| above this is "anomalous"
SMOOTH_WINDOW = 15             # frames for median smoothing
VOTE_WINDOW = 15               # frames for pose vote
POSE_STABLE_SECONDS = 1.0      # require this much stability before classifying correctness
TTS_COOLDOWN_SECONDS = 4.0     # min seconds between spoken corrections
VIDEO_SAMPLE_FPS = 2           # offline video frame sampling rate
