import asyncio
import datetime
import logging
import os
import time
from typing import Callable

from prediction_selector import PredictionItem, PredictionSelector, SavedPrediction

logger = logging.getLogger(__name__)


class PredictionSaver:
    def __init__(
        self,
        enabled: bool,
        confidence_expression: str,
        output_path: str,
        save_number_best: int = 5,
        save_number_worst: int = 5,
        number_without_label: int = 5,
        save_scan_time: float = 5.0,
        save_max_scan_time: float = 60.0,
        number_of_labels: int = 0,
        clock: Callable[[], float] = time.time,
    ):
        self.enabled = enabled
        self.output_path = output_path
        if enabled and not os.path.isdir(output_path):
            raise ValueError(f"Output path {output_path} is not a directory.")

        self.selector = PredictionSelector(
            enabled=enabled,
            confidence_expression=confidence_expression,
            save_number_best=save_number_best,
            save_number_worst=save_number_worst,
            number_without_label=number_without_label,
            save_scan_time=save_scan_time,
            save_max_scan_time=save_max_scan_time,
            clock=clock,
        )
        queue_size = self._calculate_queue_size(
            number_of_labels, save_number_best, save_number_worst, number_without_label
        )
        logger.debug(
            f"Prediction save queue size set to {queue_size} for {number_of_labels} labels."
        )
        self.queue: asyncio.Queue[SavedPrediction] = asyncio.Queue(queue_size)
        scan_time = max(0.0, float(save_scan_time))
        self.poll_interval = max(0.05, min(1.0, scan_time / 2)) if scan_time else 0.1

    @staticmethod
    def _calculate_queue_size(
        number_of_labels: int,
        save_number_best: int,
        save_number_worst: int,
        number_without_label: int,
    ) -> int:
        images_per_label = max(0, int(save_number_best)) + max(
            0, int(save_number_worst)
        )
        max_saved_images = (
            max(0, int(number_of_labels)) * images_per_label
            + max(0, int(number_without_label))
        )
        return max(32, max_saved_images)

    async def add_prediction(self, prediction: PredictionItem):
        if not self.enabled:
            logger.debug("Prediction saving is disabled. Skipping save.")
            return

        for saved in self.selector.add_prediction(prediction):
            self._enqueue(saved)

    def tick(self):
        if not self.enabled:
            return

        for saved in self.selector.tick():
            self._enqueue(saved)

    def _enqueue(self, saved: SavedPrediction):
        try:
            self.queue.put_nowait(saved)
        except asyncio.QueueFull:
            logger.warning("Prediction queue is full. Skipping save.")

    async def process(self):
        while True:
            try:
                saved = await asyncio.wait_for(
                    self.queue.get(), timeout=self.poll_interval
                )
            except asyncio.TimeoutError:
                self.tick()
                continue

            try:
                self._save(saved)
            except Exception as e:
                logger.error(f"Failed to save item: {e}")
            finally:
                self.queue.task_done()

    def _save(self, saved: SavedPrediction):
        prediction = saved.prediction
        timestamp = datetime.datetime.fromtimestamp(
            prediction.timestamp if prediction.timestamp is not None else time.time()
        )
        date = timestamp.strftime("%Y-%m-%d")
        time_string = timestamp.strftime("%H-%M-%S-%f")[:-3]

        label = self._safe_component(saved.label)
        directory = os.path.join(
            self.output_path, date, label, self._safe_component(saved.category)
        )
        os.makedirs(directory, exist_ok=True)

        detections = getattr(prediction.predictions, "predictions", None) or []
        labels = "_".join(
            sorted({f"{p.label}-{round(p.confidence * 100)}" for p in detections})
        ) or "no_labels"
        filename_base = f"{time_string}_{labels}"

        image_path = os.path.join(directory, f"{filename_base}.jpg")
        with open(image_path, "wb") as file:
            file.write(prediction.image)
        logger.info(f"Image saved to {image_path}")

        if detections:
            json_path = os.path.join(directory, f"{filename_base}.json")
            with open(json_path, "w") as file:
                file.write(prediction.predictions.model_dump_json())
            logger.info(f"Predictions saved to {json_path}")

    @staticmethod
    def _safe_component(value: str) -> str:
        return str(value).replace(os.sep, "_").replace("/", "_")
