import cv2
import torch
import torch.nn.functional as F
import numpy as np
import os
import requests

from dotenv import load_dotenv

from app.models.i3d_model import I3DModel
from app.core.config import I3D_WEIGHTS, CLASS_LIST
from app.utils.mediapipe_utils import extract_hand_status


# ================= LOAD ENV =================

load_dotenv()


# ================= CONFIG =================

clip_length = 64

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MIN_PAUSE_FRAMES = 5
MIN_SIGN_FRAMES = 10

TARGET_FPS = 25


# ================= LOAD MODEL =================

print(f"Loading I3D model on {device}...")

i3d = I3DModel(I3D_WEIGHTS, device)

with open(CLASS_LIST, "r", encoding="utf-8") as f:

    gloss_list = []

    for line in f.readlines():

        line = line.strip()

        if not line:
            continue

        # FIX:
        # "205 FEEL" -> FEEL
        parts = line.split(maxsplit=1)

        if len(parts) == 2:
            gloss = parts[1]
        else:
            gloss = parts[0]

        gloss_list.append(gloss.upper())


# ================= LOAD VIDEO =================

def load_video(video_path):

    print("Processing video...")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise Exception(f"Cannot open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)

    if original_fps <= 0:
        original_fps = 30

    print(f"Original FPS: {original_fps}")

    frame_skip = max(1, round(original_fps / TARGET_FPS))

    print(f"Frame skip: {frame_skip}")

    frames = []
    hand_flags = []

    frame_idx = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # IMPORTANT:
        # match kaggle decoding behavior
        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        resized = cv2.resize(frame, (224, 224))

        norm = (resized.astype(np.float32) / 255.0) * 2 - 1

        frames.append(norm)

        hand_flags.append(
            extract_hand_status(frame)
        )

        frame_idx += 1

        if len(frames) % 100 == 0:
            print(f"Processed {len(frames)} frames...")

    cap.release()

    frames = np.array(frames)

    print(f"Loaded {len(frames)} frames")

    return frames, hand_flags


# ================= SEGMENTATION =================

def segment_frames(frames, hand_flags):

    segments = []

    start = 0
    pause = 0

    for i in range(len(frames)):

        # TRUE = pause
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


# ================= MODEL =================

def predict_segment_topk(segment_frames, topk=5):

    if len(segment_frames) == 0:
        return []

    indices = np.linspace(
        0,
        len(segment_frames) - 1,
        clip_length
    ).astype(int)

    clip = segment_frames[indices]

    clip = clip.transpose(3, 0, 1, 2)

    clip_tensor = (
        torch.from_numpy(clip)
        .unsqueeze(0)
        .float()
        .to(device)
    )

    with torch.no_grad():

        logits = i3d(clip_tensor)

        logits = torch.mean(logits, dim=2)

        probs = F.softmax(logits, dim=1)

        top_probs, top_indices = torch.topk(
            probs,
            k=topk,
            dim=1
        )

        top_probs = top_probs.squeeze().cpu().numpy()
        top_indices = top_indices.squeeze().cpu().numpy()

    results = []

    for idx, prob in zip(top_indices, top_probs):

        gloss = gloss_list[int(idx)]

        results.append((gloss, float(prob)))

    return results


# ================= SMART SELECTION =================

def select_best_gloss(topk_results, context):

    candidates = [g for g, _ in topk_results]

    top1 = candidates[0]

    if len(context) == 0:
        return top1

    context_set = set(context)

    def score(word, is_top1=False):

        s = 0

        if word in context_set:
            s += 2

        if is_top1:
            s += 1

        return s

    best_word = top1
    best_score = score(top1, True)

    for i, word in enumerate(candidates):

        s = score(word, i == 0)

        if s > best_score:

            best_score = s
            best_word = word

    return best_word


# ================= GROQ =================

def translate_with_groq(gloss):

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return "Missing GROQ_API_KEY"

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {
                "role": "system",
                "content": "You convert ASL gloss into natural English."
            },
            {
                "role": "user",
                "content": f"""
Convert ASL gloss to English:

{gloss}

Rules:
- one sentence
- correct grammar
"""
            }
        ],
        "temperature": 0.2,
        "max_tokens": 100
    }

    try:

        r = requests.post(
            url,
            headers=headers,
            json=data
        )

        if r.status_code != 200:
            return r.text

        return r.json()["choices"][0]["message"]["content"].strip()

    except Exception as e:
        return str(e)


# ================= MAIN =================

def run_pipeline(video_path):

    print("\n" + "=" * 60)
    print("STARTING SIGN TO TEXT")
    print("=" * 60)

    frames, hand_flags = load_video(video_path)

    print("Hand flags sample:", hand_flags[:20])

    segments = segment_frames(frames, hand_flags)

    print("Segments:", segments)

    final_glosses = []

    for idx, (s, e) in enumerate(segments):

        segment_frames_data = frames[s:e]

        results = predict_segment_topk(
            segment_frames_data,
            topk=5
        )

        print(f"\n--- Segment {idx+1} ({s}-{e}) ---")

        for i, (g, p) in enumerate(results):

            print(f"{i+1}. {g} ({p:.4f})")

        if len(results) == 0:
            continue

        selected = select_best_gloss(
            results,
            final_glosses
        )

        print(f"Selected: {selected}")

        final_glosses.append(selected)

    gloss_sentence = " ".join(final_glosses)

    print("\nFINAL GLOSS:", gloss_sentence)

    english = ""

    if gloss_sentence.strip():
        english = translate_with_groq(
            gloss_sentence.lower()
        )

    print("ENGLISH:", english)

    return {
        "gloss": gloss_sentence,
        "english": english
    }