from app.models.i3d_model import I3DModel
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import os
import requests # type: ignore
from dotenv import load_dotenv
load_dotenv()   
from app.utils.mediapipe_utils import hand_detector_instance
from assets.I3D_modelfiles.pytorch_i3d import InceptionI3d
from app.core.config import I3D_WEIGHTS, CLASS_LIST
from app.models.i3d_model import I3DModel
# ================= CONFIG =================
weights_path = I3D_WEIGHTS
glosses_path = CLASS_LIST

num_classes = 300
clip_length = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MIN_PAUSE_FRAMES = 5
MIN_SIGN_FRAMES = 10


# ================= MODEL LOAD (ONCE) =================

i3d_model = I3DModel(weights_path, device)


with open(glosses_path, "r") as f:
    gloss_list = [l.strip() for l in f.readlines()]


# ================= VIDEO LOADER =================
def load_video(video_path):
    cap = cv2.VideoCapture(video_path)

    frames = []
    hand_flags = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized = cv2.resize(frame, (224, 224))
        norm = (resized.astype(np.float32) / 255.0) * 2 - 1

        frames.append(norm)

        # ✅ unified mediapipe usage
        hand_flags.append(hand_detector_instance.detect(frame))

    cap.release()

    return np.array(frames), hand_flags


# ================= SEGMENTATION =================
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


# ================= MODEL INFERENCE =================
def predict_segment(segment_frames, topk=5):
    indices = np.linspace(0, len(segment_frames) - 1, clip_length).astype(int)
    clip = segment_frames[indices]

    clip = clip.transpose(3, 0, 1, 2)
    clip_tensor = torch.from_numpy(clip).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = i3d_model.predict(clip_tensor)
        logits = torch.mean(logits, dim=2)
        probs = F.softmax(logits, dim=1)

        top_probs, top_idx = torch.topk(probs, k=topk, dim=1)

    results = []
    for i, p in zip(top_idx[0], top_probs[0]):
        results.append((gloss_list[i], float(p)))

    return results


# ================= SELECTION =================
def select_best(results, context):
    """
    Simple baseline (you can improve later with LLM or language model)
    """
    return results[0][0]


# ================= GROQ =================
def translate(gloss_sentence):

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {
                "role": "system",
                "content": "You convert ASL gloss into natural English sentences."
            },
            {
                "role": "user",
                "content": gloss_sentence
            }
        ],
        "temperature": 0.2,
        "max_tokens": 100
    }

    try:
        r = requests.post(url, headers=headers, json=data)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    except Exception as e:
        return f"Translation Error: {str(e)}"


# ================= MAIN PIPELINE =================
def run_pipeline(video_path):

    frames, hand_flags = load_video(video_path)
    segments = segment_frames(frames, hand_flags)

    final_glosses = []

    for s, e in segments:
        segment = frames[s:e]

        results = predict_segment(segment)

        selected = select_best(results, final_glosses)
        final_glosses.append(selected)

    gloss_sentence = " ".join(final_glosses)
    english = translate(gloss_sentence)

    return {
        "gloss": gloss_sentence,
        "english": english
    }