import time
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable

from confidence_evaluator import ConfidenceEvaluator
from prediction import Predictions


@dataclass
class PredictionItem:
    image: bytes
    predictions: Predictions
    forced: bool = False
    timestamp: float | None = None


class SaveCategory(StrEnum):
    BEST = "best"
    WORST = "worst"
    NO_LABEL = "no_label"


@dataclass
class SavedPrediction:
    prediction: PredictionItem
    label: str
    category: SaveCategory
    confidence: float | None = None


@dataclass
class PredictionSession:
    start_time: float
    last_trigger_time: float
    end_time: float


class PredictionSelector:
    """Selects the best/worst predictions for each label over a time window.

    A single session is started the first time any label reaches the save
    threshold. Every following image that reaches the threshold for any label
    extends the session. The session ends when no new trigger has been seen for
    ``save_scan_time``, or when ``save_max_scan_time`` has passed since the first
    trigger. When the session ends the best and worst images for every label
    detected in the window are kept, together with a number of images that have
    no labels at all.
    """

    def __init__(
        self,
        enabled: bool,
        confidence_expression: str,
        save_number_best: int,
        save_number_worst: int,
        number_without_label: int,
        save_scan_time: float,
        save_max_scan_time: float,
        clock: Callable[[], float] = time.time,
    ):
        self.enabled = enabled
        self.confidence_evaluator = ConfidenceEvaluator(confidence_expression)
        self.save_number_best = max(0, int(save_number_best))
        self.save_number_worst = max(0, int(save_number_worst))
        self.number_without_label = max(0, int(number_without_label))
        self.save_scan_time = max(0.0, float(save_scan_time))
        self.save_max_scan_time = max(self.save_scan_time, float(save_max_scan_time))
        self.clock = clock

        self.session: PredictionSession | None = None
        self.buffer: deque[PredictionItem] = deque()
        self.retention_time = self.save_max_scan_time + self.save_scan_time + 1.0

    def add_prediction(
        self, prediction: PredictionItem, now: float | None = None
    ) -> list[SavedPrediction]:
        """Record an image and return the save event if the session finished."""
        if not self.enabled:
            return []

        now = self.clock() if now is None else now
        prediction.timestamp = now

        # A trigger exactly on the current deadline extends the session instead
        # of starting a new one, so only finish a session that is strictly past
        # its deadline before applying the new triggers.
        saved = self._finalize_expired(now, include_deadline=False)

        if prediction.forced or self._triggering_labels(prediction):
            self._update_session(now)

        self.buffer.append(prediction)
        self._prune(now)
        saved.extend(self._finalize_expired(now, include_deadline=True))
        return saved

    def tick(self, now: float | None = None) -> list[SavedPrediction]:
        """Finish an expired session and drop data that can no longer be needed."""
        if not self.enabled:
            return []

        now = self.clock() if now is None else now
        saved = self._finalize_expired(now, include_deadline=True)
        self._prune(now)
        return saved

    def _triggering_labels(self, prediction: PredictionItem) -> list[str]:
        predictions = getattr(prediction.predictions, "predictions", None) or []
        labels: list[str] = []

        for item in predictions:
            if item.label in labels:
                continue
            if prediction.forced or self.confidence_evaluator.evaluate(
                item.label, item.confidence
            ):
                labels.append(item.label)

        return labels

    def _update_session(self, now: float) -> None:
        if self.session is None:
            self.session = PredictionSession(
                start_time=now,
                last_trigger_time=now,
                end_time=now + self.save_scan_time,
            )
            return

        self.session.last_trigger_time = now
        self.session.end_time = min(
            now + self.save_scan_time,
            self.session.start_time + self.save_max_scan_time,
        )

    def _finalize_expired(
        self, now: float, include_deadline: bool
    ) -> list[SavedPrediction]:
        session = self.session
        if session is None:
            return []

        expired = (
            now >= session.end_time if include_deadline else now > session.end_time
        )
        if not expired:
            return []

        self.session = None
        return self._select_session_images(session)

    def _select_session_images(self, session: PredictionSession) -> list[SavedPrediction]:
        window_start = session.start_time - self.save_scan_time
        window_end = session.end_time
        candidates = [
            item
            for item in self.buffer
            if item.timestamp is not None
            and window_start <= item.timestamp <= window_end
        ]

        labels: set[str] = set()
        without_label: list[PredictionItem] = []

        for item in candidates:
            predictions = getattr(item.predictions, "predictions", None) or []
            if not predictions:
                without_label.append(item)
                continue
            labels.update(prediction.label for prediction in predictions)

        saved: list[SavedPrediction] = []
        for label in sorted(labels):
            correct_label: list[tuple[float, PredictionItem]] = []
            for item in candidates:
                confidence = self._label_confidence(item, label)
                if confidence is not None:
                    correct_label.append((confidence, item))

            correct_label.sort(key=lambda entry: entry[0], reverse=True)
            best = correct_label[: self.save_number_best]

            remaining = correct_label[self.save_number_best :]
            remaining.sort(key=lambda entry: entry[0])
            worst = remaining[: self.save_number_worst]

            saved.extend(
                SavedPrediction(item, label, SaveCategory.BEST, confidence)
                for confidence, item in best
            )
            saved.extend(
                SavedPrediction(item, label, SaveCategory.WORST, confidence)
                for confidence, item in worst
            )

        saved.extend(
            SavedPrediction(item, "no_label", SaveCategory.NO_LABEL, None)
            for item in without_label[: self.number_without_label]
        )
        return saved

    @staticmethod
    def _label_confidence(prediction: PredictionItem, label: str) -> float | None:
        predictions = getattr(prediction.predictions, "predictions", None) or []
        confidences = [item.confidence for item in predictions if item.label == label]
        return max(confidences) if confidences else None

    def _prune(self, now: float) -> None:
        cutoff = now - self.retention_time
        while self.buffer and (self.buffer[0].timestamp or 0) < cutoff:
            self.buffer.popleft()
