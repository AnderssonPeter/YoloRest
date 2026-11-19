import numpy as np

def decode_boxes(boxes: np.ndarray, mode: str, reg_max: int = 16) -> np.ndarray:
    if mode == "distlite":
        return np.maximum(boxes, 0.0)

    if mode != "dfl":
        raise ValueError(f"Unsupported box decode mode: {mode}")

    distributions = boxes.reshape(-1, 4, reg_max)
    distributions = distributions - distributions.max(axis=2, keepdims=True)
    probabilities = np.exp(distributions)
    probabilities /= probabilities.sum(axis=2, keepdims=True)
    project = np.arange(reg_max, dtype=np.float32)
    return np.einsum("pcr,r->pc", probabilities, project)

def distances_to_xyxy(distances: np.ndarray, anchors: np.ndarray, strides: np.ndarray) -> np.ndarray:
    x1y1 = (anchors - distances[:, [0, 1]]) * strides
    x2y2 = (anchors + distances[:, [2, 3]]) * strides
    return np.concatenate((x1y1, x2y2), axis=-1)


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    boxes = boxes.copy()
    boxes[:, 2] -= boxes[:, 0]
    boxes[:, 3] -= boxes[:, 1]
    return boxes


def resize_boxes_to_original(
    boxes: np.ndarray,
    image_shape: tuple,
    model_height: int,
    model_width: int,
) -> np.ndarray:
    height, width = image_shape[:2]
    scale_x = width / model_width
    scale_y = height / model_height
    boxes = boxes.copy()
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y
    return boxes
