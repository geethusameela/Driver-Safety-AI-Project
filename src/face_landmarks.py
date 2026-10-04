import cv2
import mediapipe as mp


class FaceLandmarkDetector:

    def __init__(self):

        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process(self, frame):

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.face_mesh.process(rgb_frame)

        return results

    def get_landmarks(self, results, frame):

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]

        height, width, _ = frame.shape

        landmarks = []

        for landmark in face_landmarks.landmark:

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            landmarks.append((x, y))

        return landmarks

    def draw_landmarks(self, frame, landmarks):

        if landmarks is None:
            return frame

        for x, y in landmarks:

            cv2.circle(
                frame,
                (x, y),
                1,
                (0, 255, 0),
                -1
            )

        return frame