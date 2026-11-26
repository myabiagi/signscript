"""Stabilize consecutive static-letter predictions."""

from collections import Counter, deque

SMOOTHING_WINDOW = 6
MIN_MATCHING_PREDICTIONS = 4


def smooth_prediction(history, previous_prediction):
    """Keep the last stable letter until a new one has enough votes."""
    if len(history) < MIN_MATCHING_PREDICTIONS:
        return previous_prediction

    letter, count = Counter(history).most_common(1)[0]

    if count >= MIN_MATCHING_PREDICTIONS:
        return letter

    return previous_prediction


class PredictionSmoother:
    """Hold the last stable prediction until four votes agree."""

    def __init__(self):
        self.history = deque(maxlen=SMOOTHING_WINDOW)
        self.confidences = deque(maxlen=SMOOTHING_WINDOW)
        self.stable_prediction = "Uncertain"
        self.stable_confidence = 0.0

    def update(self, prediction, confidence=1.0):
        self.history.append(prediction)
        self.confidences.append(float(confidence))

        previous = self.stable_prediction
        self.stable_prediction = smooth_prediction(
            self.history, self.stable_prediction
        )

        if self.stable_prediction != "Uncertain":
            matching = [
                conf
                for letter, conf in zip(self.history, self.confidences)
                if letter == self.stable_prediction
            ]
            if matching:
                self.stable_confidence = sum(matching) / len(matching)
            elif self.stable_prediction != previous:
                self.stable_confidence = float(confidence)

        return self.stable_prediction, self.stable_confidence

    def reset(self):
        self.history.clear()
        self.confidences.clear()
        self.stable_prediction = "Uncertain"
        self.stable_confidence = 0.0
