from .language import (
    clean_gloss,
    map_word,
    lemmatize_words,
    llm_text_to_gloss
)
from .assets_loader import load_landmarks
from .generator import generate_frames
from .renderer import render

# ================= LOAD DATA =================
landmarks_dict = load_landmarks()

# FIX: vocab set must come from LANDMARK KEYS (like notebook)
vocab_set = set(landmarks_dict.keys())


def run_pipeline(user_sentence):
    print("INPUT:", user_sentence)

    # ================= STEP 1: LLM → GLOSS =================
    raw_gloss = llm_text_to_gloss(user_sentence)

    print("RAW GLOSS:", raw_gloss)

    # ================= STEP 2: CLEAN =================
    cleaned = clean_gloss(raw_gloss)
    print("CLEANED:", cleaned)

    # ================= STEP 3: TOKENIZE =================
    tokens = cleaned.split()
    tokens = lemmatize_words(tokens)
    print("TOKENS:", tokens)

    # ================= STEP 4: MAPPING =================
    mapped_sequence = [map_word(t, vocab_set) for t in tokens]
    print("MAPPED:", mapped_sequence)

    # ================= STEP 5: FRAME GEN =================
    frames = generate_frames(mapped_sequence)
    print("FRAMES COUNT:", len(frames))

    # ================= STEP 6: RENDER =================
    video_path = render(frames)
    print("VIDEO PATH:", video_path)

    return video_path