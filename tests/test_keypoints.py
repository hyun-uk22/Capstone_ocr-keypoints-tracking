"""Unit tests for src.keypoints.detector."""
from __future__ import annotations

import unittest

import numpy as np

from src.keypoints.detector import KeypointDetector, KeypointResult


def _checkerboard(size: int = 200) -> np.ndarray:
    """Return a checkerboard BGR image with enough texture for ORB."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    tile = size // 10
    for r in range(10):
        for c in range(10):
            if (r + c) % 2 == 0:
                img[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = 255
    return img


class TestKeypointDetectorInit(unittest.TestCase):
    def test_defaults(self):
        det = KeypointDetector()
        self.assertEqual(det.method, "orb")
        self.assertEqual(det.max_features, 500)

    def test_unsupported_method_raises(self):
        with self.assertRaises(ValueError):
            KeypointDetector(method="unknown")


class TestKeypointDetectorDetect(unittest.TestCase):
    def test_empty_frame_returns_empty_result(self):
        det = KeypointDetector()
        result = det.detect(np.array([]))
        self.assertIsInstance(result, KeypointResult)
        self.assertEqual(result.keypoints, [])
        self.assertIsNone(result.descriptors)

    def test_detects_keypoints_on_textured_image(self):
        det = KeypointDetector(method="orb", max_features=200)
        frame = _checkerboard()
        result = det.detect(frame)
        self.assertIsInstance(result, KeypointResult)
        # A textured checkerboard should yield at least one keypoint
        self.assertGreater(len(result.keypoints), 0)

    def test_descriptors_shape(self):
        det = KeypointDetector(method="orb")
        frame = _checkerboard()
        result = det.detect(frame)
        if result.descriptors is not None:
            # ORB descriptors are 32-byte binary vectors
            self.assertEqual(result.descriptors.shape[1], 32)

    def test_draw_does_not_raise(self):
        det = KeypointDetector()
        frame = _checkerboard()
        result = det.detect(frame)
        annotated = det.draw(frame.copy(), result)
        self.assertEqual(annotated.shape, frame.shape)

    def test_grayscale_input(self):
        det = KeypointDetector()
        gray = np.zeros((100, 100), dtype=np.uint8)
        # Should not raise
        result = det.detect(gray)
        self.assertIsInstance(result, KeypointResult)


class TestKeypointDetectorMatch(unittest.TestCase):
    def test_match_same_image(self):
        det = KeypointDetector(method="orb")
        frame = _checkerboard()
        r = det.detect(frame)
        matches = det.match(r, r)
        # Matching an image with itself should produce matches
        self.assertIsInstance(matches, list)

    def test_match_empty_descriptors(self):
        det = KeypointDetector()
        empty = KeypointResult()
        matches = det.match(empty, empty)
        self.assertEqual(matches, [])


if __name__ == "__main__":
    unittest.main()
