import asyncio
import datetime
import logging
import os
import cv2

from prediction import Predictions

logger = logging.getLogger(__name__)

class PredictionItem:
    def __init__(self, image: bytes, predictions: Predictions):
        self.image = image
        self.predictions = predictions

class PredictionSaver:
    def __init__(self, enabled: bool, output_path: str):
        self.enabled = enabled
        self.output_path = output_path
        if enabled and not os.path.isdir(output_path):
            raise ValueError(f"Output path {output_path} is not a directory.")
        self.queue: asyncio.Queue[PredictionItem] = asyncio.Queue(32)

    async def add_prediction(self, prediction: PredictionItem):
        if not self.enabled:
            logger.debug("Prediction saving is disabled. Skipping save.")
        elif self.queue.full():
            logger.warning("Prediction queue is full. Skipping save.")
        await self.queue.put(prediction)

    async def process(self):
        while True:
            prediction = await self.queue.get()
            try:
                timestamp = datetime.datetime.now().isoformat(timespec="seconds").replace(":", "-")
                labels = "_".join({p.label for p in prediction.predictions.predictions}) if prediction.predictions else "no_labels"
                filename_base = f"{timestamp}_{labels}"

                # Save image
                image_path = os.path.join(self.output_path, f"{filename_base}.jpg")
                cv2.imwrite(image_path, prediction.image)
                logger.info(f"Image saved to {image_path}")

                # Save predictions as JSON
                if len(prediction.predictions.predictions) > 0:
                    json_path = os.path.join(self.output_path, f"{filename_base}.json")
                    with open(json_path, "w") as file:
                        file.write(prediction.predictions.model_dump_json())
                    logger.info(f"Predictions saved to {json_path}")
            except Exception as e:
                logger.error(f"Failed to save item: {e}")
            finally:
                self.queue.task_done()