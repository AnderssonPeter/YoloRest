import logging

import cv2
import numpy as np

from YOLOFLite import YOLOFLite
from prediction import Prediction, Predictions
from yolov9_postprocess import decode_boxes, distances_to_xyxy, resize_boxes_to_original, xyxy_to_xywh

logger = logging.getLogger(__name__)


class YOLOv9OptimizedFLite(YOLOFLite):
    def __init__(
        self,
        model: str,
        labels: dict[int, str],
        conf: float = 0.25,
        iou: float = 0.45,
        device: str = "cpu",
    ):
        super().__init__(model, labels, conf, iou, device)

        self.box_decode = None
        self.reg_max = 16
        self.max_detections = 20
        self.boxes_index = None
        self.boxes_scale = 0.0
        self.boxes_zero_point = 0
        self.scores_index = None
        self.scores_scale = 0.0
        self.scores_zero_point = 0

        self.input_dtype = self.model.get_input_details()[0]["dtype"]
        details = self.model.get_output_details()
        four = [d for d in details if len(d["shape"]) == 3 and int(d["shape"][-1]) == 4]
        sixty_four = [d for d in details if len(d["shape"]) == 3 and int(d["shape"][-1]) == 64]
        boxes_detail = None
        scores_detail = None

        if sixty_four:
            boxes_detail = sixty_four[0]
            self.box_decode = "dfl"
        elif len(four) == 1:
            boxes_detail = four[0]
            self.box_decode = "distlite"
        elif len(four) == 2:
            boxes_detail, scores_detail = self._ambiguous_outputs(four)
            self.box_decode = "distlite"

        if boxes_detail is not None:
            self.boxes_index = boxes_detail["index"]
            self.boxes_scale, self.boxes_zero_point = boxes_detail["quantization"]

        if scores_detail is None:
            for detail in details:
                shape = tuple(int(v) for v in detail["shape"])
                if detail["index"] != self.boxes_index and len(shape) == 3 and shape[-1] > 1:
                    scores_detail = detail
                    break

        if scores_detail is not None:
            self.scores_index = scores_detail["index"]
            self.scores_scale, self.scores_zero_point = scores_detail["quantization"]

        if self.boxes_index is None or self.scores_index is None:
            raise ValueError(
                "Unrecognized YOLOv9 model output: expected score and box tensors."
            )

        self._generate_anchors_and_strides()

        logger.info(
            "YOLOv9 box decode: %s (boxes tensor %s, scores tensor %s)",
            self.box_decode,
            self.boxes_index,
            self.scores_index,
        )

    def _ambiguous_outputs(self, four):
        try:
            signatures = self.model.get_signature_list()
            key = next(iter(signatures))
            runner = self.model.get_signature_runner(key)
            outputs = runner.get_output_details()
            names = signatures[key]["outputs"]
            if len(names) == 2:
                logger.warning(
                    "Ambiguous 4-class outputs; using signature order: boxes=%s scores=%s",
                    names[0],
                    names[1],
                )
                return outputs[names[0]], outputs[names[1]]
        except Exception as e:
            logger.warning("Could not resolve output roles from signature: %s", e)
        logger.warning("Ambiguous outputs; assuming first is boxes and second is scores")
        return four[0], four[1]

    def _generate_anchors_and_strides(self):
        all_anchors = []
        all_strides = []
        for stride in (8, 16, 32):
            feat_h = self.in_height // stride
            feat_w = self.in_width // stride
            grid_y, grid_x = np.meshgrid(
                np.arange(feat_h, dtype=np.float32),
                np.arange(feat_w, dtype=np.float32),
                indexing="ij",
            )
            grid_coords = np.stack((grid_x.flatten(), grid_y.flatten()), axis=1)
            all_anchors.append(grid_coords + 0.5)
            all_strides.append(
                np.full((feat_h * feat_w, 1), stride, dtype=np.float32)
            )
        self.anchors = np.concatenate(all_anchors, axis=0)
        self.anchor_strides = np.concatenate(all_strides, axis=0)

    def preprocess(self, img: np.ndarray):
        img = cv2.resize(img, (self.in_width, self.in_height))
        return np.ascontiguousarray(img[..., ::-1][None]), None

    def detect(self, img: np.ndarray):
        x, _ = self.preprocess(img)
        if self.input_dtype == np.uint8:
            x = x.astype(np.uint8)
        elif self.input_dtype == np.int8:
            x = ((x.astype(np.float32) / 255.0) / self.in_scale + self.in_zero_point).astype(np.int8)
        else:
            x = x.astype(np.float32) / 255.0
        self.model.set_tensor(self.in_index, x)
        self.model.invoke()
        scores = self.model.get_tensor(self.scores_index)
        boxes = self.model.get_tensor(self.boxes_index)
        return self.postprocess(img, scores, boxes)

    def postprocess(self, img: np.ndarray, scores: np.ndarray, boxes: np.ndarray):
        scores = np.squeeze(scores, axis=0)
        boxes = np.squeeze(boxes, axis=0)
        if np.issubdtype(scores.dtype, np.integer):
            scores = (scores.astype(np.float32) - self.scores_zero_point) * self.scores_scale
        else:
            scores = scores.astype(np.float32)
        if np.issubdtype(boxes.dtype, np.integer):
            boxes = (boxes.astype(np.float32) - self.boxes_zero_point) * self.boxes_scale
        else:
            boxes = boxes.astype(np.float32)

        max_logits = scores.max(axis=1)
        confidences = 1.0 / (1.0 + np.exp(-np.clip(max_logits, -30.0, 30.0)))
        keep = confidences >= self.conf

        predictions = Predictions(predictions=[], success=True)
        if not keep.any():
            return predictions

        class_ids = scores[keep].argmax(axis=1)
        confidences = confidences[keep]
        distances = decode_boxes(boxes[keep], self.box_decode, self.reg_max)
        boxes_xyxy = distances_to_xyxy(
            distances, self.anchors[keep], self.anchor_strides[keep]
        )

        indices = cv2.dnn.NMSBoxes(
            xyxy_to_xywh(boxes_xyxy).tolist(),
            confidences.tolist(),
            self.conf,
            self.iou,
        )
        indices = np.asarray(indices).flatten()[: self.max_detections]
        if indices.size == 0:
            return predictions

        boxes_xyxy = resize_boxes_to_original(
            boxes_xyxy[indices], img.shape, self.in_height, self.in_width
        )
        for index, box in zip(indices, boxes_xyxy):
            label = self.labels.get(int(class_ids[index]))
            if label is None:
                continue
            predictions.predictions.append(
                Prediction(
                    label=label,
                    confidence=float(confidences[index]),
                    y_min=float(box[1]),
                    x_min=float(box[0]),
                    y_max=float(box[3]),
                    x_max=float(box[2]),
                )
            )
        return predictions
