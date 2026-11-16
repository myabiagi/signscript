import cv2
import mediapipe as mp

from mediapipe.tasks.python.vision import HandLandmarksConnections

# OpenCV colours use BGR format
LINE_COLOR = (246, 130, 59)
POINT_COLOR = (249, 232, 103)

def initialize_hand_detector(model_path):
    """
    Initialize the MediaPipe hand landmark detector.

    Args:
        model_path (str): Path to the pretrained hand landmark model.

    Returns:
        HandLandmarker: Initialized MediaPipe detector.
    """
    # Configure the pretrained hand landmark model
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model_path)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2
    )

    return mp.tasks.vision.HandLandmarker.create_from_options(options)

def detect_hands(frame, detector, timestamp_ms):
    """
    Detect hand landmarks in a video frame.

    Args:
        frame (np.ndarray): Webcam frame in BGR format.
        detector (HandLandmarker): Initialized MediaPipe detector.
        timestamp_ms (int): Timestamp of the current frame.

    Returns:
        HandLandmarkerResult: Detected hand landmarks.
    """
    # Convert the webcam frame from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the frame into a MediaPipe image
    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    return detector.detect_for_video(image, timestamp_ms)

def draw_landmarks(frame, result):
    """
    Draw detected hand landmarks and connections on a frame.

    Args:
        frame (np.ndarray): Webcam frame to draw on.
        result (HandLandmarkerResult): Hand detection results.

    Returns:
        np.ndarray: Frame with hand landmarks drawn.
    """
    height, width = frame.shape[:2]

    # Get MediaPipe's predefined hand connections
    connections = HandLandmarksConnections.HAND_CONNECTIONS

    for hand in result.hand_landmarks:
        points = []

        # Convert normalized coordinates into pixel coordinates
        for landmark in hand:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            points.append((x, y))

        # Draw the connections between the hand landmarks
        for connection in connections:
            start = connection.start
            end = connection.end

            cv2.line(
                frame, points[start], points[end],
                LINE_COLOR, 2
            )

        # Draw each landmark point
        for point in points:
            cv2.circle(frame, point, 4, POINT_COLOR, -1)

    return frame
