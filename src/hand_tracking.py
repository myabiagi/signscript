"""MediaPipe hand detection and landmark visualization."""

import cv2
import mediapipe as mp

from mediapipe.tasks.python.vision import HandLandmarksConnections

LINE_COLOR = (246, 130, 59)
POINT_COLOR = (249, 232, 103)

def initialize_hand_detector(model_path, running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=2):
    """Create a MediaPipe Hand Landmarker."""
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model_path)
        ),
        running_mode=running_mode,
        num_hands=num_hands,
    )

    return mp.tasks.vision.HandLandmarker.create_from_options(
        options
    )

def detect_hands(frame, detector, timestamp_ms=None):
    """Detect hand landmarks in a BGR image."""
    rgb_frame = cv2.cvtColor(
        frame, cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame,
    )

    if timestamp_ms is not None:
        return detector.detect_for_video(
            mp_image, timestamp_ms
        )

    return detector.detect(mp_image)

def draw_landmarks(frame, result):
    """Draw detected hand landmarks on the frame."""
    height, width = frame.shape[:2]

    for hand_landmarks in result.hand_landmarks:
        points = []

        for landmark in hand_landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            points.append((x, y))

        for connection in HandLandmarksConnections.HAND_CONNECTIONS:
            start = points[connection.start]
            end = points[connection.end]

            cv2.line(
                frame, start, end, LINE_COLOR, 2
            )

        for point in points:
            cv2.circle(
                frame, point, 4, POINT_COLOR, -1
            )

    return frame
