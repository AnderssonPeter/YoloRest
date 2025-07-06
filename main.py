import argparse
import asyncio
import logging
from typing import Annotated
import numpy as np
from YOLOFLite import YOLOFLite
import cv2
from prediction import Predictions
from label import parse_labels
from fastapi import FastAPI, File
from uvicorn import Config, Server

from prediction_saver import PredictionItem, PredictionSaver

if __name__ != "__main__":
    raise Exception("This script is not intended to be imported as a module.")

# Configure logging

#label_file = "models/ultralytics/yolo11n_saved_model/metadata.yaml"
#model_file = "models/ultralytics/yolo11n_saved_model/yolo11n_full_integer_quant.tflite"
#confidence_threshold = 0.25
#intersection_over_union_threshold = 0.45

parser = argparse.ArgumentParser(description="YOLO Rest Application")
parser.add_argument(
    "--log_level",
    type=str,
    default="warning",
    choices=["debug", "info", "warn", "warning", "error", "fatal", "critical"],
    help="Set the logging level (default: info)."
)
parser.add_argument("--label_file", type=str, required=True, help="Path to the label file")
parser.add_argument("--model_file", type=str, required=True, help="Path to the model file")
parser.add_argument("--device", type=str, default="cpu", help="Device to run the model on ('auto', 'cpu', 'usb', 'usb:0', 'usb:1', 'pci:1', 'pci:2')")
parser.add_argument("--confidence_threshold", type=float, default=0.25, help="Confidence threshold for detection")
parser.add_argument("--iou_threshold", type=float, default=0.45, help="Intersection over Union (IoU) threshold for detection")
parser.add_argument("--enable_save", action="store_true", help="Enable saving images and predictions")
parser.add_argument("--save_threshold", type=float, default=0.75, help="Threshold for saving predictions")
parser.add_argument("--save_path", type=str, default="./output", help="Folder to save images and predictions")

args = parser.parse_args()
log_level = logging._nameToLevel[args.log_level.upper()]

logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Log parsed arguments
logger.debug(f"Parsed arguments: {args}")

label_file = args.label_file
model_file = args.model_file
confidence_threshold = args.confidence_threshold
intersection_over_union_threshold = args.iou_threshold
device = args.device

logger.debug("Parsing labels...")
labels = parse_labels(label_file)
logger.info(f"Loaded {len(labels)} labels.")

logger.info("Initializing YOLO detector...")
detector = YOLOFLite(model_file, labels, confidence_threshold, intersection_over_union_threshold, device)
logger.info("YOLO detector initialized successfully.")

enable_save = args.enable_save
save_threshold = args.save_threshold
save_path = args.save_path

prediction_saver = PredictionSaver(enable_save, save_path)

app = FastAPI()

@app.get("/")
def root():
    logger.debug("Root endpoint accessed.")
    return {"message": "Hello World"}

@app.get("/health")
def health():
    return ""

@app.post("/detect", response_model=Predictions)
async def detect_objects(image: Annotated[bytes, File()]):
    logger.debug("Detect endpoint accessed.")
    try:
        binary_content = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(binary_content, cv2.IMREAD_COLOR)
        if img is None:
            logger.error("Failed to decode image.")
            return {"error": "Invalid image format"}
        logger.debug("Image decoded successfully. Running detection...")
        predictions = detector.detect(img)
        logger.debug(f"Detection completed. Found {len(predictions.predictions)} objects.")
        prediction_item = PredictionItem(
            image=image,
            predictions=predictions
        )
        await prediction_saver.add_prediction(prediction_item)
        return predictions
    except Exception as e:
        logger.error(f"An error occurred during detection: {e}")
        raise e

async def main():
    print("Starting application...")

    asyncio.create_task(prediction_saver.process())
    config = Config(app, host="0.0.0.0", port=8000)
    server = Server(config)
    await server.serve()

asyncio.run(main())