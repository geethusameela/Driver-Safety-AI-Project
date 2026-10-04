import cv2
import time
import joblib
import numpy as np
import pandas as pd

from collections import deque
from face_landmarks import FaceLandmarkDetector


# ============================================================
# DRIVER SAFETY AI
# REAL-TIME ML INTEGRATION
# ============================================================


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = "models/driver_safety_random_forest.pkl"

model = joblib.load(MODEL_PATH)

print("========================================")
print("DRIVER SAFETY AI")
print("========================================")
print("Random Forest model loaded successfully.")
print("Model classes:", model.classes_)


# ============================================================
# MEDIAPIPE LANDMARKS
# ============================================================

LEFT_EYE = [
    362,
    385,
    387,
    263,
    373,
    380
]

RIGHT_EYE = [
    33,
    160,
    158,
    133,
    153,
    144
]

MOUTH = [
    61,
    291,
    13,
    14,
    78,
    308
]


# ============================================================
# THRESHOLDS
# ============================================================

EAR_THRESHOLD = 0.20

CLOSED_TIME_THRESHOLD = 1.5

BLINK_MAX_DURATION = 0.5

MAR_THRESHOLD = 0.60

YAWN_TIME_THRESHOLD = 1.0

EAR_SMOOTHING = 5

HEAD_SMOOTHING = 8

HEAD_LEFT_THRESHOLD = 0.58

HEAD_RIGHT_THRESHOLD = 0.42

HEAD_DOWN_THRESHOLD = 1.18

LOOK_AWAY_TIME = 1.5


# ============================================================
# COLOURS
# ============================================================

GREEN = (0, 255, 0)

ORANGE = (0, 165, 255)

RED = (0, 0, 255)

YELLOW = (0, 255, 255)

PURPLE = (255, 0, 255)

WHITE = (255, 255, 255)


# ============================================================
# DISTANCE FUNCTION
# ============================================================

def distance(p1, p2):

    return np.linalg.norm(
        np.array(p1) - np.array(p2)
    )


# ============================================================
# EAR CALCULATION
# ============================================================

def calculate_ear(landmarks, eye):

    p1 = landmarks[eye[0]]
    p2 = landmarks[eye[1]]
    p3 = landmarks[eye[2]]
    p4 = landmarks[eye[3]]
    p5 = landmarks[eye[4]]
    p6 = landmarks[eye[5]]

    vertical_1 = distance(
        p2,
        p6
    )

    vertical_2 = distance(
        p3,
        p5
    )

    horizontal = distance(
        p1,
        p4
    )

    if horizontal == 0:

        return 0

    ear = (
        vertical_1 +
        vertical_2
    ) / (
        2.0 *
        horizontal
    )

    return ear


# ============================================================
# MAR CALCULATION
# ============================================================

def calculate_mar(landmarks):

    vertical = distance(
        landmarks[MOUTH[2]],
        landmarks[MOUTH[3]]
    )

    horizontal = distance(
        landmarks[MOUTH[0]],
        landmarks[MOUTH[1]]
    )

    if horizontal == 0:

        return 0

    mar = vertical / horizontal

    return mar


# ============================================================
# HEAD POSE
# ============================================================

def calculate_head_pose(landmarks):

    left_x = landmarks[33][0]

    right_x = landmarks[263][0]

    nose_x = landmarks[1][0]

    eye_distance = (
        right_x -
        left_x
    )

    if eye_distance == 0:

        return 0.5, 0.5

    horizontal_ratio = (
        nose_x -
        left_x
    ) / eye_distance

    nose_y = landmarks[1][1]

    left_eye_y = landmarks[33][1]

    right_eye_y = landmarks[263][1]

    eye_y = (
        left_eye_y +
        right_eye_y
    ) / 2

    face_height = abs(
        landmarks[152][1] -
        eye_y
    )

    if face_height == 0:

        vertical_ratio = 0.5

    else:

        vertical_ratio = (
            nose_y -
            eye_y
        ) / face_height

    return (
        horizontal_ratio,
        vertical_ratio
    )


# ============================================================
# RULE-BASED RISK CALCULATION
# ============================================================

def calculate_risk(
    average_ear,
    mar,
    head_status,
    looking_away_duration,
    closed_duration,
    yawn_active
):

    drowsiness_score = 0

    distraction_score = 0

    # --------------------------------------------------------
    # DROWSINESS
    # --------------------------------------------------------

    if average_ear < EAR_THRESHOLD:

        drowsiness_score += 20

    if closed_duration >= 0.5:

        drowsiness_score += 15

    if closed_duration >= 1.5:

        drowsiness_score += 30

    if mar > MAR_THRESHOLD:

        drowsiness_score += 10

    if yawn_active:

        drowsiness_score += 10

    # --------------------------------------------------------
    # DISTRACTION
    # --------------------------------------------------------

    if head_status != "FORWARD":

        distraction_score += 20

    if looking_away_duration >= 1.0:

        distraction_score += 15

    if looking_away_duration >= LOOK_AWAY_TIME:

        distraction_score += 25

    # --------------------------------------------------------
    # TOTAL RISK
    # --------------------------------------------------------

    risk_score = min(
        drowsiness_score +
        distraction_score,
        100
    )

    # --------------------------------------------------------
    # RULE STATUS
    # --------------------------------------------------------

    if risk_score >= 70:

        status = "HIGH RISK"

        status_colour = RED

    elif (
        head_status != "FORWARD"
        and
        looking_away_duration >= LOOK_AWAY_TIME
    ):

        status = "DISTRACTED"

        status_colour = RED

    elif (
        average_ear < EAR_THRESHOLD
        and
        closed_duration >= CLOSED_TIME_THRESHOLD
    ):

        status = "DROWSY"

        status_colour = RED

    elif risk_score >= 30:

        status = "CAUTION"

        status_colour = ORANGE

    else:

        status = "SAFE"

        status_colour = GREEN

    return (
        risk_score,
        status,
        status_colour
    )


# ============================================================
# DRAW TEXT
# ============================================================

def draw_text(
    frame,
    text,
    position,
    colour=WHITE,
    scale=0.65,
    thickness=2
):

    x, y = position

    # Black shadow / outline

    cv2.putText(
        frame,
        text,
        (
            x + 2,
            y + 2
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA
    )

    # Main text

    cv2.putText(
        frame,
        text,
        (
            x,
            y
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        colour,
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)


if not cap.isOpened():

    print(
        "ERROR: Could not open camera."
    )

    exit()


# ============================================================
# FACE DETECTOR
# ============================================================

detector = FaceLandmarkDetector()


# ============================================================
# SMOOTHING BUFFERS
# ============================================================

ear_history = deque(
    maxlen=EAR_SMOOTHING
)

horizontal_history = deque(
    maxlen=HEAD_SMOOTHING
)

vertical_history = deque(
    maxlen=HEAD_SMOOTHING
)


# ============================================================
# TIMERS
# ============================================================

eyes_closed_start = None

blink_start = None

looking_away_start = None

yawn_start = None


# ============================================================
# COUNTERS
# ============================================================

blink_count = 0

yawn_count = 0


# ============================================================
# STATES
# ============================================================

eyes_closed = False

yawn_active = False


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print(
            "Frame received: False"
        )

        break


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


    # ========================================================
    # FACE DETECTED
    # ========================================================

    if landmarks is not None:

        # ----------------------------------------------------
        # FULL FACE LANDMARKS
        # ----------------------------------------------------

        frame = detector.draw_landmarks(
            frame,
            landmarks
        )


        # ----------------------------------------------------
        # FACE BOUNDING BOX
        # ----------------------------------------------------

        xs = [
            point[0]
            for point in landmarks
        ]

        ys = [
            point[1]
            for point in landmarks
        ]

        x_min = max(
            min(xs) - 10,
            0
        )

        x_max = min(
            max(xs) + 10,
            frame.shape[1] - 1
        )

        y_min = max(
            min(ys) - 10,
            0
        )

        y_max = min(
            max(ys) + 10,
            frame.shape[0] - 1
        )


        cv2.rectangle(
            frame,
            (
                x_min,
                y_min
            ),
            (
                x_max,
                y_max
            ),
            GREEN,
            2
        )


        # ----------------------------------------------------
        # EYE LANDMARK POINTS
        # ----------------------------------------------------

        for index in (
            LEFT_EYE +
            RIGHT_EYE
        ):

            x, y = landmarks[index]

            cv2.circle(
                frame,
                (
                    x,
                    y
                ),
                3,
                YELLOW,
                -1
            )


        # ----------------------------------------------------
        # MOUTH LANDMARK POINTS
        # ----------------------------------------------------

        for index in MOUTH:

            x, y = landmarks[index]

            cv2.circle(
                frame,
                (
                    x,
                    y
                ),
                3,
                PURPLE,
                -1
            )


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
        ) / 2


        ear_history.append(
            average_ear
        )


        smooth_ear = np.mean(
            ear_history
        )


        # ====================================================
        # MAR
        # ====================================================

        mar = calculate_mar(
            landmarks
        )


        # ====================================================
        # HEAD POSE
        # ====================================================

        horizontal_ratio, vertical_ratio = calculate_head_pose(
            landmarks
        )


        horizontal_history.append(
            horizontal_ratio
        )

        vertical_history.append(
            vertical_ratio
        )


        smooth_horizontal = np.mean(
            horizontal_history
        )

        smooth_vertical = np.mean(
            vertical_history
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
        # CURRENT TIME
        # ====================================================

        current_time = time.time()


        # ====================================================
        # EYE STATUS / BLINK
        # ====================================================

        if smooth_ear < EAR_THRESHOLD:

            eye_status = "CLOSED"


            if not eyes_closed:

                eyes_closed = True

                eyes_closed_start = (
                    current_time
                )

                blink_start = (
                    current_time
                )


            closed_duration = (
                current_time -
                eyes_closed_start
            )


        else:

            eye_status = "OPEN"


            if eyes_closed:

                blink_duration = (
                    current_time -
                    blink_start
                )


                if (
                    blink_duration <=
                    BLINK_MAX_DURATION
                ):

                    blink_count += 1


            eyes_closed = False

            eyes_closed_start = None

            blink_start = None

            closed_duration = 0


        # ====================================================
        # YAWN DETECTION
        # ====================================================

        if mar > MAR_THRESHOLD:

            if yawn_start is None:

                yawn_start = (
                    current_time
                )


            yawn_duration = (
                current_time -
                yawn_start
            )


            if (
                yawn_duration >=
                YAWN_TIME_THRESHOLD
            ):

                if not yawn_active:

                    yawn_count += 1


                yawn_active = True


        else:

            yawn_start = None

            yawn_active = False


        # ====================================================
        # LOOK-AWAY TIMER
        # ====================================================

        if head_status != "FORWARD":

            if looking_away_start is None:

                looking_away_start = (
                    current_time
                )


            looking_away_duration = (
                current_time -
                looking_away_start
            )


        else:

            looking_away_start = None

            looking_away_duration = 0


        # ====================================================
        # MOUTH STATUS
        # ====================================================

        if mar > MAR_THRESHOLD:

            mouth_status = "OPEN"

        else:

            mouth_status = "CLOSED"


        # ====================================================
        # RULE-BASED RISK
        # ====================================================

        risk_score, rule_status, rule_colour = calculate_risk(

            smooth_ear,

            mar,

            head_status,

            looking_away_duration,

            closed_duration,

            yawn_active
        )


        # ====================================================
        # MACHINE LEARNING FEATURES
        #
        # IMPORTANT:
        # Use a pandas DataFrame so sklearn receives
        # the same feature names used during training.
        # ====================================================

        ml_features = pd.DataFrame(

            [[
                smooth_ear,
                mar,
                smooth_horizontal,
                smooth_vertical
            ]],

            columns=[
                "ear",
                "mar",
                "head_horizontal",
                "head_vertical"
            ]
        )


        # ====================================================
        # ML PREDICTION
        # ====================================================

        ml_prediction = model.predict(
            ml_features
        )[0]


        # ====================================================
        # ML PROBABILITIES
        # ====================================================

        probabilities = model.predict_proba(
            ml_features
        )[0]


        max_probability_index = np.argmax(
            probabilities
        )


        ml_confidence = (
            probabilities[
                max_probability_index
            ] * 100
        )


        # ====================================================
        # FINAL STATUS
        # ====================================================

        if ml_prediction == "DROWSY":

            final_status = "DROWSY"

            final_colour = RED


        elif ml_prediction == "DISTRACTED":

            final_status = "DISTRACTED"

            final_colour = RED


        elif risk_score >= 70:

            final_status = "HIGH RISK"

            final_colour = RED


        elif risk_score >= 30:

            final_status = "CAUTION"

            final_colour = ORANGE


        else:

            final_status = "SAFE"

            final_colour = GREEN


        # ====================================================
        # USER INTERFACE
        # ====================================================

        draw_text(
            frame,
            "DRIVER SAFETY AI",
            (
                30,
                40
            ),
            WHITE,
            0.85,
            2
        )


        draw_text(
            frame,
            f"STATUS       {final_status}",
            (
                30,
                80
            ),
            final_colour,
            0.65,
            2
        )


        draw_text(
            frame,
            f"ML PREDICTION  {ml_prediction}",
            (
                30,
                115
            ),
            final_colour,
            0.60,
            2
        )


        draw_text(
            frame,
            f"ML CONFIDENCE  {ml_confidence:.1f}%",
            (
                30,
                150
            ),
            WHITE,
            0.60,
            2
        )


        draw_text(
            frame,
            f"RISK SCORE   {risk_score}",
            (
                30,
                185
            ),
            final_colour,
            0.60,
            2
        )


        draw_text(
            frame,
            f"EAR          {smooth_ear:.3f}",
            (
                30,
                220
            ),
            WHITE,
            0.60,
            2
        )


        draw_text(
            frame,
            f"MAR          {mar:.3f}",
            (
                30,
                255
            ),
            WHITE,
            0.60,
            2
        )


        draw_text(
            frame,
            f"HEAD         {head_status}",
            (
                30,
                290
            ),
            WHITE,
            0.60,
            2
        )


        draw_text(
            frame,
            f"LOOK AWAY    {looking_away_duration:.1f}s",
            (
                30,
                325
            ),
            WHITE,
            0.60,
            2
        )


        draw_text(
            frame,
            f"EYES         {eye_status}",
            (
                30,
                360
            ),
            WHITE,
            0.60,
            2
        )


        draw_text(
            frame,
            f"MOUTH        {mouth_status}",
            (
                30,
                395
            ),
            WHITE,
            0.60,
            2
        )


        draw_text(
            frame,
            f"BLINKS       {blink_count}",
            (
                30,
                430
            ),
            WHITE,
            0.60,
            2
        )


        draw_text(
            frame,
            f"YAWNS        {yawn_count}",
            (
                30,
                465
            ),
            WHITE,
            0.60,
            2
        )


        draw_text(
            frame,
            "FACE         DETECTED",
            (
                30,
                500
            ),
            GREEN,
            0.60,
            2
        )


    # ========================================================
    # NO FACE DETECTED
    # ========================================================

    else:

        draw_text(
            frame,
            "DRIVER SAFETY AI",
            (
                30,
                40
            ),
            WHITE,
            0.85,
            2
        )


        draw_text(
            frame,
            "STATUS       NO FACE",
            (
                30,
                80
            ),
            ORANGE,
            0.65,
            2
        )


        draw_text(
            frame,
            "ML PREDICTION  --",
            (
                30,
                115
            ),
            WHITE,
            0.60,
            2
        )


        draw_text(
            frame,
            "ML CONFIDENCE  --",
            (
                30,
                150
            ),
            WHITE,
            0.60,
            2
        )


        draw_text(
            frame,
            "RISK SCORE   --",
            (
                30,
                185
            ),
            WHITE,
            0.60,
            2
        )


        draw_text(
            frame,
            "EAR          --",
            (
                30,
                220
            ),
            WHITE,
            0.60,
            2
        )


        draw_text(
            frame,
            "MAR          --",
            (
                30,
                255
            ),
            WHITE,
            0.60,
            2
        )


        draw_text(
            frame,
            "HEAD         --",
            (
                30,
                290
            ),
            WHITE,
            0.60,
            2
        )


        draw_text(
            frame,
            "LOOK AWAY    --",
            (
                30,
                325
            ),
            WHITE,
            0.60,
            2
        )


        draw_text(
            frame,
            "EYES         --",
            (
                30,
                360
            ),
            WHITE,
            0.60,
            2
        )


        draw_text(
            frame,
            "MOUTH        --",
            (
                30,
                395
            ),
            WHITE,
            0.60,
            2
        )


        draw_text(
            frame,
            f"BLINKS       {blink_count}",
            (
                30,
                430
            ),
            WHITE,
            0.60,
            2
        )


        draw_text(
            frame,
            f"YAWNS        {yawn_count}",
            (
                30,
                465
            ),
            WHITE,
            0.60,
            2
        )


        draw_text(
            frame,
            "FACE         NOT DETECTED",
            (
                30,
                500
            ),
            ORANGE,
            0.60,
            2
        )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "Driver Safety AI",
        frame
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("\nDriver Safety AI stopped.")