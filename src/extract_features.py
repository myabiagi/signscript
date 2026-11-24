import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks.python.vision import HandLandmarksConnections

LINE_COLOR = (246, 130, 59)
POINT_COLOR = (249, 232, 103)

def initialize_hand_detector(
    model_path,
    running_mode=mp.tasks.vision.RunningMode.VIDEO
):
    """
    Initialize MediaPipe for webcam frames or dataset images.

    Args:
        model_path (str or Path): Path to the pretrained model.
        running_mode (RunningMode): IMAGE or VIDEO processing mode.

    Returns:
        HandLandmarker: Initialized MediaPipe hand detector.
    """
    # Configure the hand detector
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model_path)
        ),
        running_mode=running_mode,
        num_hands=2
    )

    # Create the detector
    return mp.tasks.vision.HandLandmarker.create_from_options(options)

def detect_hands(frame, detector, timestamp_ms=None):
    """
    Detect hand landmarks in a webcam frame or dataset image.

    Args:
        frame (numpy.ndarray): OpenCV image in BGR format.
        detector (HandLandmarker): Initialized MediaPipe detector.
        timestamp_ms (int, optional): Timestamp for VIDEO mode.

    Returns:
        HandLandmarkerResult: Detected hand landmarks.
    """
    # Convert the image from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the image into MediaPipe's format
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect hands in continuous video frames
    if timestamp_ms is not None:
        return detector.detect_for_video(mp_image, timestamp_ms)

    # Detect hands in independent dataset images
    return detector.detect(mp_image)

def draw_landmarks(frame, result):
    """
    Draw detected hand landmarks and connections on a frame.

    Args:
        frame (numpy.ndarray): OpenCV image in BGR format.
        result (HandLandmarkerResult): Detected hand landmarks.

    Returns:
        numpy.ndarray: Frame with landmarks drawn.
    """
    height, width = frame.shape[:2]

    # Process each detected hand
    for hand_landmarks in result.hand_landmarks:
        points = []

        # Convert normalized landmarks to pixel coordinates
        for landmark in hand_landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            points.append((x, y))

        # Draw connections between landmarks
        for connection in HandLandmarksConnections.HAND_CONNECTIONS:
            start = points[connection.start]
            end = points[connection.end]

            cv2.line(frame, start, end, LINE_COLOR, 2)

        # Draw landmark points
        for point in points:
            cv2.circle(frame, point, 4, POINT_COLOR, -1)

    return frame

def normalize_landmarks(hand_landmarks, handedness):
    """
    Normalize hand landmarks for position, size, and handedness.

    Args:
        hand_landmarks (list): MediaPipe's 21 detected landmarks.
        handedness (str): Detected hand label ("Left" or "Right").

    Returns:
        numpy.ndarray: Flattened array of 63 normalized coordinates.
    """
    # Validate the landmarks and handedness
    if len(hand_landmarks) != 21:
        raise ValueError("Expected 21 hand landmarks.")

    if handedness not in ("Left", "Right"):
        raise ValueError(f"Unexpected handedness label: {handedness}")

    # Convert the 21 landmarks into a NumPy array
    points = []

    for landmark in hand_landmarks:
        points.append([landmark.x, landmark.y, landmark.z])

    points = np.array(points, dtype=np.float32)

    # Move the wrist to the origin
    wrist = points[0].copy()
    points = points - wrist

    # Mirror left-labelled hands to a common orientation
    if handedness == "Left":
        points[:, 0] = -points[:, 0]

    # Find the maximum distance from the wrist
    distances = np.linalg.norm(points, axis=1)
    max_distance = np.max(distances)

    # Normalize the hand size
    if max_distance > 0:
        points = points / max_distance

    # Convert 21 x 3 coordinates into 63 features
    return points.flatten()
