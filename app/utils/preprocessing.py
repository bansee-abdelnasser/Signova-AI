import numpy as np

def sample_clip(frames, clip_length=64):
    idx = np.linspace(0, len(frames)-1, clip_length).astype(int)
    return frames[idx]