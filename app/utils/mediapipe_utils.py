import mediapipe as mp
import cv2

class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands # type: ignore
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2
        )

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            return False   # hand active (signing)
        return True        # pause / no hand


# GLOBAL INSTANCE (VERY IMPORTANT)
hand_detector_instance = HandDetector()