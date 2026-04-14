"""Unit tests for src.ocr.detector."""
from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.ocr.detector import OCRDetector, OCRResult


class TestOCRResult(unittest.TestCase):
    def test_fields(self):
        r = OCRResult(bbox=[[0, 0], [10, 0], [10, 5], [0, 5]], text="hello", confidence=0.9)
        self.assertEqual(r.text, "hello")
        self.assertAlmostEqual(r.confidence, 0.9)


class TestOCRDetectorInit(unittest.TestCase):
    def test_defaults(self):
        det = OCRDetector()
        self.assertEqual(det.languages, ["en"])
        self.assertFalse(det.gpu)
        self.assertAlmostEqual(det.min_confidence, 0.5)

    def test_custom_params(self):
        det = OCRDetector(languages=["ko", "en"], gpu=False, min_confidence=0.7)
        self.assertEqual(det.languages, ["ko", "en"])
        self.assertAlmostEqual(det.min_confidence, 0.7)


class TestOCRDetectorDetect(unittest.TestCase):
    """Tests for OCRDetector.detect() with a mocked EasyOCR reader."""

    def _make_detector_with_mock_reader(self, raw_output):
        det = OCRDetector(min_confidence=0.5)
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = raw_output
        det._reader = mock_reader
        return det

    def _sample_frame(self):
        return np.zeros((100, 100, 3), dtype=np.uint8)

    def test_returns_empty_for_empty_frame(self):
        det = OCRDetector()
        result = det.detect(np.array([]))
        self.assertEqual(result, [])

    def test_filters_low_confidence(self):
        raw = [
            ([[0, 0], [10, 0], [10, 5], [0, 5]], "hello", 0.9),
            ([[0, 10], [10, 10], [10, 15], [0, 15]], "world", 0.3),
        ]
        det = self._make_detector_with_mock_reader(raw)
        results = det.detect(self._sample_frame())
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].text, "hello")

    def test_returns_all_above_threshold(self):
        raw = [
            ([[0, 0], [10, 0], [10, 5], [0, 5]], "A", 0.8),
            ([[0, 10], [10, 10], [10, 15], [0, 15]], "B", 0.6),
        ]
        det = self._make_detector_with_mock_reader(raw)
        results = det.detect(self._sample_frame())
        self.assertEqual(len(results), 2)

    def test_draw_does_not_raise(self):
        raw = [([[0, 0], [50, 0], [50, 20], [0, 20]], "Test", 0.95)]
        det = self._make_detector_with_mock_reader(raw)
        frame = self._sample_frame()
        ocr_results = det.detect(frame)
        annotated = det.draw(frame.copy(), ocr_results)
        self.assertEqual(annotated.shape, frame.shape)


if __name__ == "__main__":
    unittest.main()
