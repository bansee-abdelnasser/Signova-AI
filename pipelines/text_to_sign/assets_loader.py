import json
import numpy as np
from .config import JSON_PATH, LANDMARKS_PATH


def load_landmarks():
    with open(JSON_PATH) as f:
        data = json.load(f)

    landmarks_data = np.load(LANDMARKS_PATH, allow_pickle=True)
    landmarks_array = landmarks_data["landmarks"]

    landmarks_dict = {}

    for sample, lm in zip(data, landmarks_array):
        gloss = sample["gloss"].lower().strip()
        landmarks_dict[gloss] = lm

    return landmarks_dict