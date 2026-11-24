import unittest
from dataclasses import dataclass, field

from prediction_selector import PredictionItem, PredictionSelector, SaveCategory


@dataclass
class FakePrediction:
    label: str
    confidence: float


@dataclass
class FakePredictions:
    predictions: list[FakePrediction] = field(default_factory=list)


def make_item(
    detections: dict[str, float] | None = None,
    forced: bool = False,
    image: bytes = b"image",
) -> PredictionItem:
    predictions = [
        FakePrediction(label, confidence)
        for label, confidence in (detections or {}).items()
    ]
    return PredictionItem(
        image=image, predictions=FakePredictions(predictions), forced=forced
    )


def make_selector(
    best: int = 2,
    worst: int = 2,
    without_label: int = 1,
    scan_time: float = 2.0,
    max_scan_time: float = 10.0,
    threshold: str = "0.75",
    enabled: bool = True,
) -> PredictionSelector:
    return PredictionSelector(
        enabled=enabled,
        confidence_expression=threshold,
        save_number_best=best,
        save_number_worst=worst,
        number_without_label=without_label,
        save_scan_time=scan_time,
        save_max_scan_time=max_scan_time,
    )


def timestamps(results, category=None, label=None):
    return [
        result.prediction.timestamp
        for result in results
        if (category is None or result.category == category)
        and (label is None or result.label == label)
    ]


class TestPredictionSelector(unittest.TestCase):
    def test_selects_best_and_worst_correct_label_images(self):
        selector = make_selector(best=2, worst=2, without_label=0)

        selector.add_prediction(make_item({"person": 0.80}), now=0)
        selector.add_prediction(make_item({"person": 0.95}), now=1)
        selector.add_prediction(make_item({"person": 0.76}), now=2)
        selector.add_prediction(make_item({"person": 0.99}), now=3)
        selector.add_prediction(make_item({"person": 0.77}), now=4)

        self.assertEqual(selector.tick(5.9), [])

        results = selector.tick(6)
        self.assertEqual(timestamps(results, "best"), [3, 1])
        self.assertEqual(timestamps(results, "worst"), [2, 4])
        self.assertTrue(all(result.label == "person" for result in results))
        self.assertIsInstance(results[0].category, SaveCategory)

    def test_session_looks_back_before_first_trigger(self):
        selector = make_selector(best=1, worst=1, without_label=1, scan_time=3)

        selector.add_prediction(make_item({"person": 0.50}), now=0)
        selector.add_prediction(make_item(None), now=1)
        self.assertEqual(selector.add_prediction(make_item({"person": 0.90}), now=2), [])

        results = selector.tick(5)
        self.assertEqual(timestamps(results, "best"), [2])
        self.assertEqual(timestamps(results, "worst"), [0])
        self.assertEqual(timestamps(results, "no_label"), [1])

    def test_lookback_does_not_reach_past_scan_time(self):
        selector = make_selector(best=1, worst=1, without_label=0, scan_time=2)

        selector.add_prediction(make_item({"person": 0.50}), now=0)
        selector.add_prediction(make_item({"person": 0.90}), now=5)

        results = selector.tick(7)
        self.assertEqual(timestamps(results, "best"), [5])
        self.assertEqual(timestamps(results, "worst"), [])

    def test_new_trigger_extends_the_timer(self):
        selector = make_selector(best=1, worst=1, without_label=0, scan_time=2)

        selector.add_prediction(make_item({"person": 0.90}), now=0)
        selector.add_prediction(make_item({"person": 0.80}), now=1.5)
        selector.add_prediction(make_item({"person": 0.50}), now=2.5)

        self.assertEqual(selector.tick(3.4), [])

        results = selector.tick(3.5)
        self.assertEqual(timestamps(results, "best"), [0])
        self.assertEqual(timestamps(results, "worst"), [2.5])

    def test_max_scan_time_caps_session_length(self):
        selector = make_selector(best=1, worst=0, without_label=0, scan_time=2, max_scan_time=5)

        selector.add_prediction(make_item({"person": 0.80}), now=0)
        selector.add_prediction(make_item({"person": 0.90}), now=2)
        selector.add_prediction(make_item({"person": 0.95}), now=4)

        self.assertEqual(selector.tick(4.9), [])

        results = selector.tick(5)
        self.assertEqual(timestamps(results, "best"), [4])
        self.assertIsNone(selector.session)

    def test_no_more_images_ends_session_after_scan_time(self):
        selector = make_selector(best=1, worst=0, without_label=0, scan_time=2)

        selector.add_prediction(make_item({"person": 0.90}), now=0)

        self.assertEqual(selector.tick(1.99), [])
        self.assertEqual(timestamps(selector.tick(2), "best"), [0])

    def test_saves_limited_number_of_images_without_labels(self):
        selector = make_selector(best=1, worst=0, without_label=2, scan_time=2)

        selector.add_prediction(make_item({"person": 0.90}), now=0)
        selector.add_prediction(make_item(None), now=0.5)
        selector.add_prediction(make_item(None), now=1)
        selector.add_prediction(make_item(None), now=1.5)

        results = selector.tick(2)
        self.assertEqual(timestamps(results, "no_label"), [0.5, 1])

    def test_single_timer_is_extended_by_any_label(self):
        selector = make_selector(best=1, worst=0, without_label=0, scan_time=2)

        selector.add_prediction(make_item({"person": 0.90}), now=0)
        selector.add_prediction(make_item({"deer": 0.90}), now=1)

        self.assertEqual(selector.tick(2.5), [])
        self.assertIsNotNone(selector.session)

        results = selector.tick(3)
        self.assertEqual(timestamps(results, "best", "person"), [0])
        self.assertEqual(timestamps(results, "best", "deer"), [1])
        self.assertIsNone(selector.session)

    def test_each_detected_label_gets_its_own_best_and_worst(self):
        selector = make_selector(best=1, worst=1, without_label=0, scan_time=5)

        selector.add_prediction(make_item({"person": 0.90}), now=0)
        selector.add_prediction(make_item({"deer": 0.10}), now=1)
        selector.add_prediction(make_item({"person": 0.80}), now=2)
        selector.add_prediction(make_item({"person": 0.60}), now=3)

        results = selector.tick(8)
        self.assertEqual(timestamps(results, "best", "person"), [0])
        self.assertEqual(timestamps(results, "worst", "person"), [3])
        self.assertEqual(timestamps(results, "best", "deer"), [1])
        self.assertEqual(timestamps(results, "worst", "deer"), [])

    def test_forced_item_starts_a_session_regardless_of_threshold(self):
        selector = make_selector(
            best=1, worst=0, without_label=0, scan_time=2, threshold="0.90"
        )

        selector.add_prediction(make_item({"person": 0.10}, forced=True), now=0)

        results = selector.tick(2)
        self.assertEqual(timestamps(results, "best"), [0])

    def test_forced_item_without_labels_is_saved_as_no_label(self):
        selector = make_selector(best=0, worst=0, without_label=1, scan_time=2)

        selector.add_prediction(make_item(None, forced=True), now=0)

        results = selector.tick(2)
        self.assertEqual(timestamps(results, "no_label"), [0])

    def test_images_below_threshold_do_not_start_a_session(self):
        selector = make_selector(best=1, worst=1, without_label=1, scan_time=2)

        selector.add_prediction(make_item({"person": 0.50}), now=0)
        selector.add_prediction(make_item(None), now=1)

        self.assertEqual(selector.tick(100), [])
        self.assertIsNone(selector.session)

    def test_best_and_worst_never_contain_the_same_image(self):
        selector = make_selector(best=2, worst=2, without_label=0, scan_time=2)

        selector.add_prediction(make_item({"person": 0.90}), now=0)
        selector.add_prediction(make_item({"person": 0.80}), now=1)
        selector.add_prediction(make_item({"person": 0.70}), now=2)

        results = selector.tick(4)
        best = timestamps(results, "best")
        worst = timestamps(results, "worst")
        self.assertEqual(best, [0, 1])
        self.assertEqual(worst, [2])
        self.assertTrue(set(best).isdisjoint(worst))

    def test_trigger_exactly_at_deadline_extends_session(self):
        selector = make_selector(best=1, worst=1, without_label=0, scan_time=2)

        selector.add_prediction(make_item({"person": 0.90}), now=0)
        self.assertEqual(selector.add_prediction(make_item({"person": 0.95}), now=2), [])

        self.assertEqual(selector.tick(3.9), [])

        results = selector.tick(4)
        self.assertEqual(timestamps(results, "best"), [2])
        self.assertEqual(timestamps(results, "worst"), [0])

    def test_trigger_after_deadline_finishes_old_session_and_starts_new_one(self):
        selector = make_selector(best=1, worst=0, without_label=0, scan_time=2)

        selector.add_prediction(make_item({"person": 0.90}), now=0)
        saved = selector.add_prediction(make_item({"person": 0.80}), now=2.01)

        self.assertEqual(timestamps(saved, "best"), [0])
        self.assertIsNotNone(selector.session)
        self.assertEqual(selector.session.start_time, 2.01)

        results = selector.tick(4.01)
        self.assertEqual(timestamps(results, "best"), [2.01])

    def test_trigger_at_max_deadline_finalizes_session(self):
        selector = make_selector(
            best=1, worst=1, without_label=0, scan_time=2, max_scan_time=4
        )

        selector.add_prediction(make_item({"person": 0.80}), now=0)
        selector.add_prediction(make_item({"person": 0.90}), now=2)

        results = selector.add_prediction(make_item({"person": 0.95}), now=4)
        self.assertEqual(timestamps(results, "best"), [4])
        self.assertEqual(timestamps(results, "worst"), [0])
        self.assertIsNone(selector.session)

    def test_second_tick_after_save_returns_nothing(self):
        selector = make_selector(best=1, worst=0, without_label=0, scan_time=2)

        selector.add_prediction(make_item({"person": 0.90}), now=0)

        self.assertEqual(len(selector.tick(2)), 1)
        self.assertEqual(selector.tick(100), [])

    def test_multiple_labels_in_one_image_are_ranked_separately(self):
        selector = make_selector(best=1, worst=0, without_label=0, scan_time=2)

        selector.add_prediction(make_item({"person": 0.90, "deer": 0.80}), now=0)
        selector.add_prediction(make_item({"person": 0.95, "deer": 0.70}), now=1)

        results = selector.tick(3)
        self.assertEqual(timestamps(results, "best", "person"), [1])
        self.assertEqual(timestamps(results, "best", "deer"), [0])

    def test_highest_confidence_is_used_for_duplicate_labels_in_one_image(self):
        selector = make_selector(best=1, worst=0, without_label=0, scan_time=2)
        item = PredictionItem(
            b"image",
            FakePredictions(
                [FakePrediction("person", 0.76), FakePrediction("person", 0.99)]
            ),
        )

        selector.add_prediction(item, now=0)
        selector.add_prediction(make_item({"person": 0.90}), now=1)

        results = selector.tick(3)
        self.assertEqual(timestamps(results, "best"), [0])
        self.assertEqual(results[0].confidence, 0.99)

    def test_scan_time_zero_saves_immediately(self):
        selector = make_selector(
            best=1, worst=0, without_label=0, scan_time=0, max_scan_time=10
        )

        results = selector.add_prediction(make_item({"person": 0.90}), now=0)

        self.assertEqual(timestamps(results, "best"), [0])
        self.assertIsNone(selector.session)

    def test_negative_counts_are_clamped_to_zero(self):
        selector = make_selector(best=-1, worst=-1, without_label=-1, scan_time=2)

        selector.add_prediction(make_item({"person": 0.90}), now=0)
        selector.add_prediction(make_item(None), now=0.5)

        self.assertEqual(selector.tick(2), [])

    def test_unlabeled_images_are_not_ranked_as_labels(self):
        selector = make_selector(best=1, worst=1, without_label=1, scan_time=2)

        selector.add_prediction(make_item({"person": 0.90}), now=0)
        selector.add_prediction(make_item(None), now=0.5)

        results = selector.tick(2)
        self.assertEqual(timestamps(results, "best", "person"), [0])
        self.assertEqual(timestamps(results, "worst", "person"), [])
        self.assertEqual(timestamps(results, "no_label"), [0.5])

    def test_label_specific_threshold_controls_session_start(self):
        selector = make_selector(
            best=1,
            worst=0,
            without_label=0,
            scan_time=2,
            threshold="person:0.90,0.50",
        )

        selector.add_prediction(make_item({"person": 0.85}), now=0)
        self.assertIsNone(selector.session)

        selector.add_prediction(make_item({"deer": 0.60}), now=1)
        self.assertIsNotNone(selector.session)

        selector.add_prediction(make_item({"person": 0.95}), now=2)

        results = selector.tick(4)
        self.assertEqual(timestamps(results, "best", "person"), [2])
        self.assertEqual(timestamps(results, "best", "deer"), [1])

    def test_disabled_selector_does_not_save_anything(self):
        selector = make_selector(
            best=1, worst=1, without_label=1, scan_time=2, enabled=False
        )

        selector.add_prediction(make_item({"person": 0.90}), now=0)

        self.assertEqual(selector.tick(100), [])


if __name__ == "__main__":
    unittest.main()
