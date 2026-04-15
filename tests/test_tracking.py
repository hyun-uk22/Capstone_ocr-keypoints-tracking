"""Unit tests for src.tracking.tracker."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.tracking.tracker import ObjectTracker, TrackResult


def _frame(h: int = 100, w: int = 100) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestObjectTrackerInit(unittest.TestCase):
    def test_defaults(self):
        t = ObjectTracker()
        self.assertEqual(t.tracker_type, "csrt")

    def test_unsupported_type_raises(self):
        with self.assertRaises(ValueError):
            ObjectTracker(tracker_type="unknown")

    def test_active_ids_initially_empty(self):
        t = ObjectTracker()
        self.assertEqual(t.active_ids, [])


class TestObjectTrackerOperations(unittest.TestCase):
    """Tests using a mocked underlying OpenCV tracker."""

    def _make_tracker(self, tracker_type: str = "csrt") -> ObjectTracker:
        t = ObjectTracker(tracker_type=tracker_type)
        return t

    def _mock_cv_tracker(self, success: bool = True, bbox=(10, 20, 30, 40)):
        mock = MagicMock()
        mock.init.return_value = True
        mock.update.return_value = (success, bbox)
        return mock

    def test_init_registers_track_id(self):
        t = self._make_tracker()
        mock_cv = self._mock_cv_tracker()
        with patch.object(t, "_create_tracker", return_value=mock_cv):
            tid = t.init(_frame(), (0, 0, 20, 20))
        self.assertEqual(tid, 0)
        self.assertIn(0, t.active_ids)

    def test_multiple_inits_unique_ids(self):
        t = self._make_tracker()
        mock1 = self._mock_cv_tracker()
        mock2 = self._mock_cv_tracker()
        with patch.object(t, "_create_tracker", side_effect=[mock1, mock2]):
            id1 = t.init(_frame(), (0, 0, 10, 10))
            id2 = t.init(_frame(), (50, 50, 10, 10))
        self.assertNotEqual(id1, id2)

    def test_update_returns_results(self):
        t = self._make_tracker()
        mock_cv = self._mock_cv_tracker(success=True, bbox=(5, 5, 20, 20))
        with patch.object(t, "_create_tracker", return_value=mock_cv):
            t.init(_frame(), (0, 0, 20, 20))
        results = t.update(_frame())
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].success)
        self.assertEqual(results[0].bbox, (5, 5, 20, 20))

    def test_remove_clears_track(self):
        t = self._make_tracker()
        mock_cv = self._mock_cv_tracker()
        with patch.object(t, "_create_tracker", return_value=mock_cv):
            tid = t.init(_frame(), (0, 0, 20, 20))
        t.remove(tid)
        self.assertNotIn(tid, t.active_ids)

    def test_clear_removes_all(self):
        t = self._make_tracker()
        mock1 = self._mock_cv_tracker()
        mock2 = self._mock_cv_tracker()
        with patch.object(t, "_create_tracker", side_effect=[mock1, mock2]):
            t.init(_frame(), (0, 0, 10, 10))
            t.init(_frame(), (50, 50, 10, 10))
        t.clear()
        self.assertEqual(t.active_ids, [])

    def test_draw_does_not_raise(self):
        t = self._make_tracker()
        results = [TrackResult(track_id=0, bbox=(10, 10, 30, 30), success=True)]
        frame = _frame()
        annotated = t.draw(frame.copy(), results)
        self.assertEqual(annotated.shape, frame.shape)


if __name__ == "__main__":
    unittest.main()
