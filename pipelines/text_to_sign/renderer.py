import cv2
import numpy as np
from .config import WIDTH, HEIGHT, FPS, POSE_CONNECTIONS, FINGER_COLORS

# =========================
# HAND
# =========================
def draw_hand_stylish(img, hand_points, color_map, thickness=4):
    h, w = img.shape[:2]

    pts = [(int(x*w), int(y*h)) for x,y,_ in hand_points]

    fingers = {
        "thumb":[0,1,2,3,4],
        "index":[0,5,6,7,8],
        "middle":[0,9,10,11,12],
        "ring":[0,13,14,15,16],
        "pinky":[0,17,18,19,20]
    }

    for name, idxs in fingers.items():
        color = color_map[name]

        for i in range(len(idxs)-1):
            x1,y1 = pts[idxs[i]]
            x2,y2 = pts[idxs[i+1]]
            cv2.line(img,(x1,y1),(x2,y2),color,thickness,cv2.LINE_AA)


# =========================
# POSE
# =========================
def draw_pose_lines(img, points, connections, color=(0,0,255), thickness=4):
    h, w = img.shape[:2]

    pts = [(int(x*w), int(y*h)) for x,y,_ in points]

    for i,j in connections:
        if i < len(pts) and j < len(pts):
            cv2.line(img, pts[i], pts[j], color, thickness, cv2.LINE_AA)


# =========================
# FACE / LANDMARKS
# =========================
def draw_points_and_connections(img, points, color=(255,255,255)):
    h, w = img.shape[:2]

    pts = [(int(x*w), int(y*h)) for x,y,_ in points]

    for (x,y) in pts:
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(img, (x,y), 1, color, -1)


# =========================
# MAIN DRAW (MATCH NOTEBOOK)
# =========================
def draw_skeleton(frame, img):
    draw_pose_lines(img, frame[42:75], POSE_CONNECTIONS, (0,0,255), 4)

    draw_hand_stylish(img, frame[0:21], FINGER_COLORS, 4)
    draw_hand_stylish(img, frame[21:42], FINGER_COLORS, 4)

    # FACE 
    draw_points_and_connections(img, frame[75:553], (0,0,255))


# =========================
# RENDER FUNCTION (KEEP THIS)
# =========================
def render(frames, output="output.mp4"):
    writer = cv2.VideoWriter(
        output,
        cv2.VideoWriter_fourcc(*"mp4v"), # pyright: ignore[reportAttributeAccessIssue]
        FPS,
        (WIDTH, HEIGHT)
    )

    for frame in frames:
        img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        draw_skeleton(frame, img)
        writer.write(img)

    writer.release()
    return output