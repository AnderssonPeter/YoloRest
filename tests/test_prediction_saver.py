import unittest

from prediction_saver import PredictionSaver


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


if __name__ == "__main__":
    unittest.main()
