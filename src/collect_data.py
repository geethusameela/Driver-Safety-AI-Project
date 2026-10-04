import cv2
import csv
import os
import time
import uuid
from collections import deque

import numpy as np

from face_landmarks import FaceLandmarkDetector


# ============================================================
# CONFIGURATION
# ============================================================

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

MOUTH = [61, 291, 13, 14, 78, 308]

EAR_THRESHOLD = 0.20
MAR_THRESHOLD = 0.60

HEAD_LEFT_THRESHOLD = 0.58
HEAD_RIGHT_THRESHOLD = 0.42
HEAD_DOWN_THRESHOLD = 1.18

EAR_SMOOTHING = 5
HEAD_SMOOTHING = 8


# ============================================================
# OUTPUT FILE
# ============================================================

DATA_FOLDER = "data"

os.makedirs(DATA_FOLDER, exist_ok=True)

CSV_FILE = os.path.join(
    DATA_FOLDER,
    "driver_temporal_features.csv"
)


# ============================================================
# CREATE SESSION ID
# ============================================================

session_id = str(uuid.uuid4())[:8]

print("\n========================================")
print("DRIVER SAFETY AI - TEMPORAL DATA")
print("========================================")

print("\nSession ID:", session_id)

print("\nControls:")
print("1 = ALERT")
print("2 = DROWSY")
print("3 = DISTRACTED")
print("Q = QUIT")


# ============================================================
# CREATE CSV FILE
# ============================================================

file_exists = os.path.isfile(CSV_FILE)

csv_file = open(
    CSV_FILE,
    "a",
    newline=""
)

writer = csv.writer(csv_file)

if not file_exists:

    writer.writerow([
        "session_id",
        "frame",
        "timestamp",
        "ear",
        "mar",
        "head_horizontal",
        "head_vertical",
        "head_status",
        "eye_status",
        "mouth_status",
        "label"
    ])


# ============================================================
# INITIALISE CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open camera.")

    csv_file.close()

    raise SystemExit


# ============================================================
# INITIALISE FACE DETECTOR
# ============================================================

detector = FaceLandmarkDetector()


# ============================================================
# SMOOTHING BUFFERS
# ============================================================

ear_history = deque(
    maxlen=EAR_SMOOTHING
)

head_history = deque(
    maxlen=HEAD_SMOOTHING
)


# ============================================================
# VARIABLES
# ============================================================

current_label = "UNLABELLED"

frame_number = 0

start_time = time.time()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def distance(p1, p2):

    return np.linalg.norm(
        np.array(p1) - np.array(p2)
    )


def calculate_ear(landmarks, eye_points):

    p1 = landmarks[eye_points[0]]
    p2 = landmarks[eye_points[1]]
    p3 = landmarks[eye_points[2]]
    p4 = landmarks[eye_points[3]]
    p5 = landmarks[eye_points[4]]
    p6 = landmarks[eye_points[5]]

    vertical_1 = distance(p2, p6)
    vertical_2 = distance(p3, p5)

    horizontal = distance(p1, p4)

    if horizontal == 0:

        return 0

    ear = (
        vertical_1 + vertical_2
    ) / (
        2.0 * horizontal
    )

    return ear


def calculate_mar(landmarks):

    top = landmarks[MOUTH[2]]
    bottom = landmarks[MOUTH[3]]

    left = landmarks[MOUTH[0]]
    right = landmarks[MOUTH[1]]

    vertical = distance(
        top,
        bottom
    )

    horizontal = distance(
        left,
        right
    )

    if horizontal == 0:

        return 0

    return vertical / horizontal


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("ERROR: Could not read camera frame.")

        break


    frame_number += 1


    # --------------------------------------------------------
    # MIRROR CAMERA
    # --------------------------------------------------------

    frame = cv2.flip(
        frame,
        1
    )


    # --------------------------------------------------------
    # PROCESS FACE
    # --------------------------------------------------------

    results = detector.process(
        frame
    )

    landmarks = detector.get_landmarks(
        results,
        frame
    )


    if landmarks is not None:

        # ====================================================
        # EAR
        # ====================================================

        left_ear = calculate_ear(
            landmarks,
            LEFT_EYE
        )

        right_ear = calculate_ear(
            landmarks,
            RIGHT_EYE
        )

        average_ear = (
            left_ear +
            right_ear
        ) / 2.0


        # ====================================================
        # MAR
        # ====================================================

        mar = calculate_mar(
            landmarks
        )


        # ====================================================
        # SMOOTH EAR
        # ====================================================

        ear_history.append(
            average_ear
        )

        smooth_ear = np.mean(
            ear_history
        )


        # ====================================================
        # HEAD POSITION
        # ====================================================

        forehead = landmarks[10]

        nose = landmarks[1]

        left_face = landmarks[234]

        right_face = landmarks[454]

        face_width = distance(
            left_face,
            right_face
        )

        face_height = distance(
            landmarks[10],
            landmarks[152]
        )


        if face_width == 0:

            horizontal_position = 0

        else:

            horizontal_position = (
                nose[0] - forehead[0]
            ) / face_width


        if face_height == 0:

            vertical_position = 0

        else:

            vertical_position = (
                nose[1] - forehead[1]
            ) / face_height


        head_history.append(
            (
                horizontal_position,
                vertical_position
            )
        )


        smooth_horizontal = np.mean(
            [
                item[0]
                for item in head_history
            ]
        )

        smooth_vertical = np.mean(
            [
                item[1]
                for item in head_history
            ]
        )


        # ====================================================
        # HEAD STATUS
        # ====================================================

        if smooth_horizontal < HEAD_RIGHT_THRESHOLD:

            head_status = "LEFT"

        elif smooth_horizontal > HEAD_LEFT_THRESHOLD:

            head_status = "RIGHT"

        elif smooth_vertical > HEAD_DOWN_THRESHOLD:

            head_status = "DOWN"

        else:

            head_status = "FORWARD"


        # ====================================================
        # EYE STATUS
        # ====================================================

        if smooth_ear < EAR_THRESHOLD:

            eye_status = "CLOSED"

        else:

            eye_status = "OPEN"


        # ====================================================
        # MOUTH STATUS
        # ====================================================

        if mar > MAR_THRESHOLD:

            mouth_status = "OPEN"

        else:

            mouth_status = "NORMAL"


        # ====================================================
        # DRAW LANDMARKS
        # ====================================================

        frame = detector.draw_landmarks(
            frame,
            landmarks
        )


        # ====================================================
        # SAVE DATA
        # ====================================================

        timestamp = time.time()

        writer.writerow([
            session_id,
            frame_number,
            timestamp,
            round(smooth_ear, 4),
            round(mar, 4),
            round(smooth_horizontal, 4),
            round(smooth_vertical, 4),
            head_status,
            eye_status,
            mouth_status,
            current_label
        ])

        csv_file.flush()


        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.putText(
            frame,
            f"LABEL: {current_label}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"EAR: {smooth_ear:.3f}",
            (30, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"MAR: {mar:.3f}",
            (30, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"HEAD: {head_status}",
            (30, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"FRAME: {frame_number}",
            (30, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


    else:

        cv2.putText(
            frame,
            "FACE NOT DETECTED",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # ========================================================
    # SHOW CAMERA
    # ========================================================

    cv2.imshow(
        "Temporal Data Collection",
        frame
    )


    # ========================================================
    # KEYBOARD CONTROLS
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("1"):

        current_label = "ALERT"

        print(
            "\nLabel changed to: ALERT"
        )


    elif key == ord("2"):

        current_label = "DROWSY"

        print(
            "\nLabel changed to: DROWSY"
        )


    elif key == ord("3"):

        current_label = "DISTRACTED"

        print(
            "\nLabel changed to: DISTRACTED"
        )


    elif key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

csv_file.close()


print("\n========================================")
print("DATA COLLECTION FINISHED")
print("========================================")

print("\nSession ID:", session_id)

print("Frames collected:", frame_number)

print(
    "\nSaved to:",
    CSV_FILE
)