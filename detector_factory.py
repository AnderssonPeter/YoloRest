from YOLOFLite import YOLOFLite
from YOLOv9OptimizedFLite import YOLOv9OptimizedFLite


def create_detector(
    model_type: str,
    model_file: str,
    labels: dict[int, str],
    confidence_threshold: float,
    iou_threshold: float,
    device: str,
):
    if model_type == "optimized_yolov9":
        return YOLOv9OptimizedFLite(
            model_file, labels, confidence_threshold, iou_threshold, device
        )
    if model_type == "yolov8":
        return YOLOFLite(
            model_file, labels, confidence_threshold, iou_threshold, device
        )
    raise ValueError(f"Unsupported model type: {model_type}")
