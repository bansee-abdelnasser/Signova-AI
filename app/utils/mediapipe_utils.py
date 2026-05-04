import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2
)

HAND_LOWER_THRESHOLD = 0.85


def extract_hand_status(frame):
    """
    EXACT SAME LOGIC AS NOTEBOOK

    Returns:
        True  -> pause / hands lowered
        False -> sign active
    """

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            for lm in hand.landmark:

                # hand raised -> signing
                if lm.y < HAND_LOWER_THRESHOLD:
                    return False

    # all hands lowered
    return True