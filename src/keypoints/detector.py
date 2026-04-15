"""
Keypoint detector module.

Detects and describes local keypoints in an image using OpenCV feature
detectors (ORB by default; SIFT is available when
``opencv-contrib-python`` is installed).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class KeypointResult:
    """Holds keypoints and their descriptors for a single frame."""

    keypoints: List[cv2.KeyPoint] = field(default_factory=list)
    descriptors: np.ndarray | None = None


class KeypointDetector:
    """Detect keypoints and compute descriptors.

    Parameters
    ----------
    method:
        Feature detector to use.  Supported values: ``'orb'`` (default)
        and ``'sift'`` (requires ``opencv-contrib-python``).
    max_features:
        Maximum number of features to retain (default ``500``).
    """

    _SUPPORTED = ("orb", "sift")

    def __init__(
        self,
        method: str = "orb",
        max_features: int = 500,
    ) -> None:
        method = method.lower()
        if method not in self._SUPPORTED:
            raise ValueError(
                f"Unsupported method '{method}'. Choose one of {self._SUPPORTED}."
            )
        self.method = method
        self.max_features = max_features
        self._detector = self._build_detector()
        self._matcher = self._build_matcher()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> KeypointResult:
        """Detect keypoints and compute descriptors for *frame*.

        Parameters
        ----------
        frame:
            BGR or grayscale image.

        Returns
        -------
        KeypointResult
        """
        if frame is None or frame.size == 0:
            return KeypointResult()

        gray = self._to_gray(frame)
        kps, descs = self._detector.detectAndCompute(gray, None)
        return KeypointResult(keypoints=list(kps), descriptors=descs)

    def match(
        self,
        result_a: KeypointResult,
        result_b: KeypointResult,
        max_distance: float = 50.0,
    ) -> List[cv2.DMatch]:
        """Match keypoints between two frames.

        Parameters
        ----------
        result_a, result_b:
            Keypoint results from :meth:`detect`.
        max_distance:
            Maximum descriptor distance; matches above this threshold
            are discarded.

        Returns
        -------
        list[cv2.DMatch]
            Sorted list of good matches.
        """
        if result_a.descriptors is None or result_b.descriptors is None:
            return []

        matches = self._matcher.match(result_a.descriptors, result_b.descriptors)
        good = [m for m in matches if m.distance < max_distance]
        return sorted(good, key=lambda m: m.distance)

    def draw(
        self,
        frame: np.ndarray,
        result: KeypointResult,
        color: Tuple[int, int, int] = (0, 0, 255),
    ) -> np.ndarray:
        """Draw detected keypoints onto *frame*.

        Parameters
        ----------
        frame:
            Source image.
        result:
            Keypoint result from :meth:`detect`.
        color:
            BGR colour for the drawn keypoints.

        Returns
        -------
        np.ndarray
            Annotated image.
        """
        return cv2.drawKeypoints(
            frame,
            result.keypoints,
            None,
            color=color,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_detector(self):
        if self.method == "orb":
            return cv2.ORB_create(nfeatures=self.max_features)
        # SIFT (requires opencv-contrib-python)
        return cv2.SIFT_create(nfeatures=self.max_features)

    def _build_matcher(self):
        if self.method == "orb":
            return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        return cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
