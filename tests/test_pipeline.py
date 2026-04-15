"""Unit tests for src.pipeline."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np

from src.keypoints.detector import KeypointResult
from src.pipeline import Pipeline, PipelineResult


def _frame() -> np.ndarray:
    return np.zeros((100, 100, 3), dtype=np.uint8)


class TestPipelineInit(unittest.TestCase):
    def test_empty_pipeline(self):
        p = Pipeline()
        self.assertIsNone(p.ocr)
        self.assertIsNone(p.kp)
        self.assertIsNone(p.tracker)

    def test_frame_index_starts_at_zero(self):
        p = Pipeline()
        self.assertEqual(p._frame_index, 0)


class TestPipelineProcess(unittest.TestCase):
    def _make_mock_ocr(self, results=None):
        m = MagicMock()
        m.detect.return_value = results or []
        return m

    def _make_mock_kp(self, kp_result=None):
        m = MagicMock()
        m.detect.return_value = kp_result or KeypointResult()
        return m

    def _make_mock_tracker(self, track_results=None):
        m = MagicMock()
        m.update.return_value = track_results or []
        return m

    def test_process_increments_frame_index(self):
        p = Pipeline()
        p.process(_frame())
        p.process(_frame())
        self.assertEqual(p._frame_index, 2)

    def test_process_calls_all_submodules(self):
        ocr = self._make_mock_ocr()
        kp = self._make_mock_kp()
        tracker = self._make_mock_tracker()
        p = Pipeline(ocr_detector=ocr, keypoint_detector=kp, object_tracker=tracker)
        p.process(_frame())
        ocr.detect.assert_called_once()
        kp.detect.assert_called_once()
        tracker.update.assert_called_once()

    def test_process_skips_disabled_submodules(self):
        ocr = self._make_mock_ocr()
        p = Pipeline(ocr_detector=ocr, keypoint_detector=None, object_tracker=None)
        p.process(_frame())
        ocr.detect.assert_called_once()

    def test_result_type(self):
        p = Pipeline()
        result = p.process(_frame())
        self.assertIsInstance(result, PipelineResult)
        self.assertEqual(result.frame_index, 0)

    def test_reset_clears_frame_index(self):
        p = Pipeline()
        p.process(_frame())
        p.process(_frame())
        p.reset()
        self.assertEqual(p._frame_index, 0)

    def test_reset_clears_tracker(self):
        tracker = self._make_mock_tracker()
        p = Pipeline(object_tracker=tracker)
        p.reset()
        tracker.clear.assert_called_once()


if __name__ == "__main__":
    unittest.main()
