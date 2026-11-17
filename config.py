
from pathlib import Path


# ==============================
# PROJECT PATH
# ==============================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "pose_landmarker_full.task"


# ==============================
# CAMERA SETTINGS
# ==============================

CAMERA_INDEX = 0 # Usually we used :               0 = default webcam
                                                  #1 = second camera
                                                  #2 = third camera

FRAME_WIDTH = 1280 #This requests a 1280×720 camera image.
FRAME_HEIGHT = 720


# ==============================
# MEDIAPIPE SETTINGS
# ==============================

NUM_POSES = 1

MIN_POSE_DETECTION_CONFIDENCE = 0.5

MIN_POSE_PRESENCE_CONFIDENCE = 0.5

MIN_TRACKING_CONFIDENCE = 0.5