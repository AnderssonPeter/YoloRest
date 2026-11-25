import datetime
import os
import tempfile
import unittest
from dataclasses import dataclass, field

from prediction_saver import PredictionSaver
from prediction_selector import (
    PredictionItem,
    SaveCategory,
    SavedPrediction,
)


@dataclass
class FakePrediction:
    label: str
    confidence: float


@dataclass
class FakePredictions:
    predictions: list[FakePrediction] = field(default_factory=list)

    def model_dump_json(self):
        return "{}"


def make_saver(**overrides) -> PredictionSaver:
    arguments = {
        "enabled": False,
        "confidence_expression": "0.75",
        "output_path": "/tmp",
        "save_number_best": 5,
        "save_number_worst": 5,
        "number_without_label": 5,
        "save_scan_time": 5.0,
        "save_max_scan_time": 60.0,
        "number_of_labels": 0,
    }
    arguments.update(overrides)
    return PredictionSaver(**arguments)


class TestPredictionSaverQueueSize(unittest.TestCase):
    def test_queue_grows_with_number_of_labels(self):
        saver = make_saver(number_of_labels=100)

        self.assertEqual(saver.queue.maxsize, 100 * (5 + 5) + 5)

    def test_queue_size_includes_all_labels_and_unlabeled_images(self):
        saver = make_saver(
            number_of_labels=20,
            save_number_best=2,
            save_number_worst=3,
            number_without_label=2,
        )

        self.assertEqual(saver.queue.maxsize, 20 * (2 + 3) + 2)

    def test_queue_has_a_minimum_size(self):
        saver = make_saver(
            number_of_labels=1,
            save_number_best=2,
            save_number_worst=2,
            number_without_label=1,
        )

        self.assertEqual(saver.queue.maxsize, 32)

    def test_queue_handles_no_labels(self):
        saver = make_saver(number_of_labels=0)

        self.assertEqual(saver.queue.maxsize, 32)

    def test_negative_values_are_clamped(self):
        saver = make_saver(
            number_of_labels=-10,
            save_number_best=-1,
            save_number_worst=-1,
            number_without_label=-1,
        )

        self.assertEqual(saver.queue.maxsize, 32)


class TestPredictionSaverOutputPath(unittest.TestCase):
    def test_saves_under_date_time_range_and_category(self):
        start = datetime.datetime(2023, 11, 14, 15, 30, 0).timestamp()
        end = start + 10
        prediction = PredictionItem(
            b"image",
            FakePredictions([FakePrediction("person", 0.90)]),
            timestamp=start + 2,
        )
        saved = SavedPrediction(
            prediction,
            "person",
            SaveCategory.BEST,
            0.90,
            start,
            end,
        )

        with tempfile.TemporaryDirectory() as output_path:
            saver = make_saver(output_path=output_path)
            saver._save(saved)

            start_time = datetime.datetime.fromtimestamp(start)
            end_time = datetime.datetime.fromtimestamp(end)
            expected_directory = os.path.join(
                output_path,
                start_time.strftime("%Y-%m-%d"),
                f"{start_time.strftime('%H.%M.%S')}-{end_time.strftime('%H.%M.%S')}",
                "best",
            )

            files = os.listdir(expected_directory)
            self.assertTrue(any(name.endswith(".jpg") for name in files))
            self.assertTrue(any(name.endswith(".json") for name in files))
            self.assertIn("15.30.00-15.30.10", expected_directory)
            self.assertNotIn(":", expected_directory)

    def test_background_category_uses_background_directory(self):
        timestamp = datetime.datetime(2023, 11, 14, 8, 5, 0).timestamp()
        prediction = PredictionItem(
            b"image", FakePredictions(), timestamp=timestamp
        )
        saved = SavedPrediction(
            prediction,
            "background",
            SaveCategory.BACKGROUND,
            None,
            timestamp,
            timestamp + 10,
        )

        with tempfile.TemporaryDirectory() as output_path:
            saver = make_saver(output_path=output_path)
            saver._save(saved)

            start_time = datetime.datetime.fromtimestamp(timestamp)
            expected_directory = os.path.join(
                output_path,
                start_time.strftime("%Y-%m-%d"),
                f"{start_time.strftime('%H.%M.%S')}-"
                f"{datetime.datetime.fromtimestamp(timestamp + 10).strftime('%H.%M.%S')}",
                "background",
            )

            files = os.listdir(expected_directory)
            self.assertTrue(any(name.endswith(".jpg") for name in files))
            self.assertIn("08.05.00-08.05.10", expected_directory)
            self.assertNotIn(":", expected_directory)


if __name__ == "__main__":
    unittest.main()
