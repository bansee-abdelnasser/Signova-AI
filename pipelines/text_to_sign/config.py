import os

JSON_PATH = "assets\\text_to_sign\\filtered_one_video_per_gloss.json"
LANDMARKS_PATH = "assets\\text_to_sign\\wlasl2000_filtered_keypoints\\filtered_landmarks_V3\\filtered_landmarks_V3.npz"

FPS = 25
WIDTH = 1280
HEIGHT = 720

POSE_CONNECTIONS = [
    (11,13),(13,15),(12,14),(14,16),(11,12),
    (23,24),(11,23),(12,24),(23,25),(24,26),
    (25,27),(26,28)
]

FINGER_COLORS = {
    "thumb": (0,128,255),
    "index": (0,255,0),
    "middle": (255,255,0),
    "ring": (255,0,0),
    "pinky": (255,0,255)
}
