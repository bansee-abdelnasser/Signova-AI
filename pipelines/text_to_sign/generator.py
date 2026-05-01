import re

import numpy as np
from .assets_loader import load_landmarks

landmarks_dict = load_landmarks()

def generate_frames(mapped_sequence):

    all_frames = []

    for item in mapped_sequence:

        if item['type'] == 'direct':
            word = item['mapped'].lower()

            if word in landmarks_dict:
                frames = [np.array(f, dtype=float) for f in landmarks_dict[word]]
                all_frames.extend(frames)

            else:
                print(f"⚠️ Word '{word}' not found")

        else:
            word = re.sub(r'[^a-z]', '', item['original'].lower())
            spelled = list(word)

            for letter in spelled:
                if letter in landmarks_dict:
                    frames = [np.array(f, dtype=float) for f in landmarks_dict[letter]]
                    all_frames.extend(frames)

                else:
                    print(f"⚠️ Letter '{letter}' not found")

    return all_frames