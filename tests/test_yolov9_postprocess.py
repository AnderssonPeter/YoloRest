import unittest

import numpy as np

from yolov9_postprocess import (
    decode_boxes,
    distances_to_xyxy,
    resize_boxes_to_original,
    xyxy_to_xywh,
)


class TestYolov9Postprocess(unittest.TestCase):

    def test_decode_boxes_distlite_clamps_negative_values(self):
        boxes = np.array([[-1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        result = decode_boxes(boxes, "distlite")
        np.testing.assert_allclose(result, [[0.0, 2.0, 3.0, 4.0]])

    def test_decode_boxes_dfl_returns_expected_expectation(self):
        boxes = np.zeros((1, 64), dtype=np.float32)
        result = decode_boxes(boxes, "dfl")
        np.testing.assert_allclose(result, [[7.5, 7.5, 7.5, 7.5]])

    def test_distances_to_xyxy_uses_anchors_and_strides(self):
        distances = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        anchors = np.array([[10.0, 10.0]], dtype=np.float32)
        strides = np.array([[8.0]], dtype=np.float32)
        result = distances_to_xyxy(distances, anchors, strides)
        np.testing.assert_allclose(result, [[72.0, 64.0, 104.0, 112.0]])

    def test_xyxy_to_xywh(self):
        boxes = np.array([[10.0, 20.0, 30.0, 50.0]], dtype=np.float32)
        result = xyxy_to_xywh(boxes)
        np.testing.assert_allclose(result, [[10.0, 20.0, 20.0, 30.0]])

    def test_resize_boxes_to_original_uses_squash_scaling(self):
        boxes = np.array([[0.0, 0.0, 320.0, 320.0]], dtype=np.float32)
        result = resize_boxes_to_original(boxes, (480, 640, 3), 320, 320)
        np.testing.assert_allclose(result, [[0.0, 0.0, 640.0, 480.0]])


if __name__ == "__main__":
    unittest.main()
