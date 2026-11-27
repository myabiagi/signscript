"""
Shared motion / commit thresholds for desktop prep and the web demo.
"""

# Motion vs idle (normalized landmark units / frame, EMA-smoothed)
MOVE_SPEED = 0.022
IDLE_SPEED = 0.012
SPEED_EMA = 0.35

# Capture buffer
MAX_BUFFER_FRAMES = 90
MIN_STROKE_FRAMES = 4  # tuned for fast webcam J/Z
MIN_BUFFER_FRAMES = 8

# Stroke geometry (wrist-relative tip path)
MIN_RELATIVE_TIP_PATH = 0.28
MIN_TIP_OVER_WRIST_RATIO = 1.35
MIN_DIRECTION_CHANGES = 1

# Accept J/Z from the dynamic model
MIN_DYNAMIC_CONFIDENCE = 0.45

# Drop idle head/tail before classifying a stroke
TRIM_SPEED = 0.002

# After a J/Z commit, ignore static briefly (browser)
DYNAMIC_HOLD_MS = 1800

# Static display smoothing (browser)
PREDICT_EVERY_MS = 120
STABLE_FRAMES = 4

# Landmark indices
WRIST_INDEX = 0
INDEX_TIP = 8
PINKY_TIP = 20
TIP_INDICES = (4, 8, 12, 16, 20)


def as_web_config():
    """JSON-friendly dict for GET /api/config (camelCase for JS)."""
    return {
        "moveSpeed": MOVE_SPEED,
        "idleSpeed": IDLE_SPEED,
        "speedEma": SPEED_EMA,
        "maxBufferFrames": MAX_BUFFER_FRAMES,
        "minStrokeFrames": MIN_STROKE_FRAMES,
        "minRelativeTipPath": MIN_RELATIVE_TIP_PATH,
        "minTipOverWristRatio": MIN_TIP_OVER_WRIST_RATIO,
        "minDirectionChanges": MIN_DIRECTION_CHANGES,
        "minDynamicConfidence": MIN_DYNAMIC_CONFIDENCE,
        "dynamicHoldMs": DYNAMIC_HOLD_MS,
        "predictEveryMs": PREDICT_EVERY_MS,
        "stableFrames": STABLE_FRAMES,
        "tipIndices": list(TIP_INDICES),
        "wristIndex": WRIST_INDEX,
        "indexTip": INDEX_TIP,
        "pinkyTip": PINKY_TIP,
    }
