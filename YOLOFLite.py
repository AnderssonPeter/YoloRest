# Based on the code from Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
import logging
import os
import time
from typing import Tuple, Union, Dict, List
import cv2
import numpy as np
import yaml

from tflite_runtime.interpreter import Interpreter, load_delegate

from detection import Detection
from label import Label

logger = logging.getLogger(__name__)

class YOLOFLite:
    """
    A class for performing object detection using the YOLOv8 model with TensorFlow Lite.

    This class handles model loading, preprocessing, inference, and visualization of detection results.

    Attributes:
        model (Interpreter): TensorFlow Lite interpreter for the YOLOv8 model.
        labels (Dict[int, Label]): Dictionary mapping class IDs to Label names.
        conf (float): Confidence threshold for filtering detections.
        iou (float): Intersection over Union threshold for non-maximum suppression.
        color_palette (np.ndarray): Random color palette for visualization with shape (num_labels, 3).
        in_width (int): Input width required by the model.
        in_height (int): Input height required by the model.
        in_index (int): Input tensor index in the model.
        in_scale (float): Input quantization scale factor.
        in_zero_point (int): Input quantization zero point.
        int8 (bool): Whether the model uses int8 quantization.
        out_index (int): Output tensor index in the model.
        out_scale (float): Output quantization scale factor.
        out_zero_point (int): Output quantization zero point.

    Methods:
        letterbox: Resizes and pads image while maintaining aspect ratio.
        draw_detections: Draws bounding boxes and labels on the input image.
        preprocess: Preprocesses the input image before inference.
        postprocess: Processes model outputs to extract and visualize detections.
        detect: Performs object detection on an input image.
    """

    # todo: change metadata to labels, that is a dict int, string, support both yaml and labelmap https://github.com/google-coral/tflite/blob/eced31ac01e9c2636150decef7d3c335d0feb304/python/examples/classification/classify_image.py#L55
    def __init__(self, model: str, labels: Dict[int, Label], conf: float = 0.25, iou: float = 0.45, device: str = "cpu"):
        """
        Initialize an instance of the YOLOv8TFLite class.

        Args:
            model (str): Path to the TFLite model file.
            labels: (Dict[int, Label]):  Dictionary mapping class IDs to Label names.
            conf (float): Confidence threshold for filtering detections.
            iou (float): IoU threshold for non-maximum suppression.
            device (str): auto, cpu, usb, usb:0, usb:1, pci:1, pci:2
        """
        self.conf = conf
        self.iou = iou
        self.labels = labels

        np.random.seed(42)  # Set seed for reproducible colors
        self.color_palette = np.random.uniform(128, 255, size=(len(self.labels), 3))

        logger.info(f"Attempting to load TPU as {device}")
        if device == "cpu":
            self.model = Interpreter(model_path=model)
        else:
            try:
                # Initialize the TFLite interpreter
                device_config = {"device": device}
                interpreter_delegate = load_delegate("libedgetpu.so.1.0", device_config)
                logger.info("TPU found")
                self.model = Interpreter(
                    model_path=model,
                    experimental_delegates=[interpreter_delegate],
                )
            except ValueError:
                _, ext = os.path.splitext(model)

                if ext and ext != ".tflite":
                    logger.error(
                        "Incorrect model used with EdgeTPU. Only .tflite models can be used with a Coral EdgeTPU or CPU."
                    )
                else:
                    logger.error(
                        "No EdgeTPU was detected. If you do not have a Coral device yet, you must configure CPU detectors."
                    )

                raise

        self.model.allocate_tensors()

        # Get input details
        input_details = self.model.get_input_details()[0]
        self.in_width, self.in_height = input_details["shape"][1:3]
        self.in_index = input_details["index"]
        self.in_scale, self.in_zero_point = input_details["quantization"]
        self.int8 = input_details["dtype"] == np.int8

        # Get output details
        output_details = self.model.get_output_details()[0]
        self.out_index = output_details["index"]
        self.out_scale, self.out_zero_point = output_details["quantization"]

    def letterbox(
        self, img: np.ndarray, new_shape: Tuple[int, int] = (640, 640)
    ) -> Tuple[np.ndarray, Tuple[float, float]]:
        """
        Resize and pad image while maintaining aspect ratio.

        Args:
            img (np.ndarray): Input image with shape (H, W, C).
            new_shape (Tuple[int, int]): Target shape (height, width).

        Returns:
            (np.ndarray): Resized and padded image.
            (Tuple[float, float]): Padding ratios (top/height, left/width) for coordinate adjustment.
        """
        start_time = time.time()
        shape = img.shape[:2]  # Current shape [height, width]

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding

        if shape[::-1] != new_unpad:  # Resize if needed
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        duration_ms = (time.time() - start_time) * 1000
        logger.debug(f"letterbox took {duration_ms:.2f} ms")

        return img, (top / img.shape[0], left / img.shape[1])

    def draw_detections(self, img: np.ndarray, detections: List[Detection]) -> None:
        """
        Draw bounding boxes and labels on the input image based on the detected objects.

        Args:
            img (np.ndarray): The input image to draw detections on.
            detections: List[Detection]: List of detected objects with their bounding boxes, scores, and class IDs.
        """
        font_scale = 10
        for detection in detections:
            color = self.color_palette[detection.label.id]
            cv2.rectangle(img, (int(detection.left), int(detection.top)), (int(detection.right), int(detection.bottom)), color, 2)
            label = f"{detection.label.name}: {detection.score:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            label_x = detection.left
            label_y = detection.top - 10 if detection.top - 10 > label_height else detection.top + 10
            cv2.rectangle(
                img,
                (int(label_x), int(label_y - label_height)),
                (int(label_x + label_width), int(label_y + label_height)),
                color,
                cv2.FILLED,
            )

            # Draw text
            cv2.putText(img, label, (int(label_x), int(label_y)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
        """
        Preprocess the input image before performing inference.

        Args:
            img (np.ndarray): The input image to be preprocessed with shape (H, W, C).

        Returns:
            (np.ndarray): Preprocessed image ready for model input.
            (Tuple[float, float]): Padding ratios for coordinate adjustment.
        """
        img, pad = self.letterbox(img, (self.in_width, self.in_height))
        img = img[..., ::-1][None]  # BGR to RGB and add batch dimension (N, H, W, C) for TFLite
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32)
        return img / 255, pad  # Normalize to [0, 1]

    def postprocess(self, img: np.ndarray, outputs: np.ndarray, pad: Tuple[float, float]) -> List[Detection]:
        """
        Process model outputs to extract and visualize detections.

        Args:
            img (np.ndarray): The original input image.
            outputs (np.ndarray): Raw model outputs.
            pad (Tuple[float, float]): Padding ratios from preprocessing.

        Returns:
            (np.ndarray): The input image with detections drawn on it.
        """
        # Adjust coordinates based on padding and scale to original image size
        outputs[:, 0] -= pad[1]
        outputs[:, 1] -= pad[0]
        outputs[:, :4] *= max(img.shape)

        # Transform outputs to [x, y, w, h] format
        outputs = outputs.transpose(0, 2, 1)
        outputs[..., 0] -= outputs[..., 2] / 2  # x center to top-left x
        outputs[..., 1] -= outputs[..., 3] / 2  # y center to top-left y
        detections = []

        for out in outputs:
            # Get scores and apply confidence threshold
            scores = out[:, 4:].max(-1)
            keep = scores > self.conf
            if not keep.any():
                logger.debug("No detections passed the confidence threshold.")
                continue
            boxes = out[keep, :4]
            scores = scores[keep]
            class_ids = out[keep, 4:].argmax(-1)
            # Apply non-maximum suppression
            indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf, self.iou).flatten()
            for i in indices:
                label = self.labels[class_ids[i]]
                score = scores[i]
                box = boxes[i]
                left, top, w, h = box
                right = left + w
                bottom = top + h
                detections.append(Detection(label=label, score=score, top=top, left=left, bottom=bottom, right=right))
        return detections

    def detect(self, img: np.ndarray) -> List[Detection]:
        """
        Perform object detection on an input image.

        Args:
            img (np.ndarray): Image

        Returns:
            (List[Detection]): List of detected objects with their bounding boxes, scores and label.
        """
        # Load and preprocess image
        
        x, pad = self.preprocess(img)

        # Apply quantization if model is int8
        if self.int8:
            x = (x / self.in_scale + self.in_zero_point).astype(np.int8)

        # Set input tensor and run inference
        start_time = time.time()
        self.model.set_tensor(self.in_index, x)
        self.model.invoke()
        duration_ms = (time.time() - start_time) * 1000
        logger.debug(f"detect took {duration_ms:.2f} ms")

        # Get output and dequantize if necessary
        y = self.model.get_tensor(self.out_index)
        if self.int8:
            y = (y.astype(np.float32) - self.out_zero_point) * self.out_scale

        # Process detections and return result
        return self.postprocess(img, y, pad)