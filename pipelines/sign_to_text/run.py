import cv2
import torch
import torch.nn.functional as F
import numpy as np
import os

from app.core.config import I3D_WEIGHTS, CLASS_LIST
from app.models.i3d_model import I3DModel
from app.utils.mediapipe_utils import hand_detector_instance

# ---------------- CONFIG ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clip_length = 64
MIN_PAUSE_FRAMES = 5
MIN_SIGN_FRAMES = 10

# ---------------- MODEL ----------------
print("Loading I3D model...")

i3d_model = I3DModel(I3D_WEIGHTS, device)

with open(CLASS_LIST, "r") as f:
    gloss_list = [line.strip().upper() for line in f.readlines()]


# ---------------- VIDEO LOADING ----------------
def load_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frames = []
    hand_flags = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

       #print("FRAME SHAPE:", frame.shape)

        resized = cv2.resize(frame, (224, 224))
        norm = (resized.astype(np.float32) / 255.0) * 2 - 1

        frames.append(norm)
        hand_flags.append(hand_detector_instance.detect(frame))

    cap.release()

    frames = np.array(frames)
    print("TOTAL FRAMES READ:", len(frames))

    return frames, hand_flags


# ---------------- SEGMENTATION ----------------
def segment_frames(frames, hand_flags):
    segments = []
    start = 0
    pause = 0

    for i in range(len(frames)):
        if hand_flags[i]:
            pause += 1
        else:
            if pause >= MIN_PAUSE_FRAMES:
                end = i - pause
                if end - start >= MIN_SIGN_FRAMES:
                    segments.append((start, end))
                start = i
            pause = 0

    if len(frames) - start >= MIN_SIGN_FRAMES:
        segments.append((start, len(frames) - 1))

    return segments


# ---------------- INFERENCE ----------------
def predict_segment(segment_frames, topk=5):

    if len(segment_frames) < clip_length:
        return []

    results_accum = {}

    # 🔥 FIX 1: proper sliding window (NOT naive chunking)
    stride = clip_length // 2

    for start in range(0, len(segment_frames) - clip_length, stride):

        clip = segment_frames[start:start + clip_length]

        # safety check
        if clip.shape[0] != clip_length:
            continue

        clip = clip.transpose(3, 0, 1, 2)
        clip_tensor = torch.from_numpy(clip).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = i3d_model(clip_tensor)
            logits = torch.mean(logits, dim=2)
            probs = F.softmax(logits, dim=1)

            top_probs, top_idx = torch.topk(probs, k=topk, dim=1)

        # 🔥 FIX 2: accumulate scores properly
        for i, p in zip(top_idx[0], top_probs[0]):
            idx = int(i)
            if idx < len(gloss_list):
                word = gloss_list[idx]

                if word not in results_accum:
                    results_accum[word] = 0

                results_accum[word] += float(p)

    if not results_accum:
        return []

    # 🔥 FIX 3: sort final results
    sorted_results = sorted(results_accum.items(), key=lambda x: x[1], reverse=True)

    return sorted_results[:topk]


# ---------------- SELECTION ----------------
def select_best(results, context):
    if not results:
        return ""

    return results[0][0]


# ---------------- PIPELINE ----------------
def run_pipeline(video_path):

    print("\nSTARTING PIPELINE")

    frames, hand_flags = load_video(video_path)

    print("HAND FLAGS SAMPLE:", hand_flags[:20])
    print("TRUE COUNT:", sum(hand_flags))

    segments = segment_frames(frames, hand_flags)

    print("SEGMENTS:", segments)

    final_glosses = []

    for s, e in segments:
        segment = frames[s:e]

        results = predict_segment(segment)

        if not results:
            continue

        selected = select_best(results, final_glosses)
        final_glosses.append(selected)

    gloss_sentence = " ".join(final_glosses)

    print("\nGLOSS:", gloss_sentence)

    # IMPORTANT: no API test yet
    english = "API NOT RUN (testing logic only)"

    print("ENGLISH:", english)

    return {
        "gloss": gloss_sentence,
        "english": english
    }