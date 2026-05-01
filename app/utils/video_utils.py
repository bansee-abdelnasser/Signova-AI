#unused file
import cv2
import numpy as np

def load_video(path, size=(224,224)):
    cap = cv2.VideoCapture(path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, size)
        frame = (frame.astype(np.float32) / 255.0) * 2 - 1
        frames.append(frame)

    cap.release()
    return np.array(frames)