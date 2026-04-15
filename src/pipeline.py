"""
Unified pipeline that combines OCR, keypoint detection, and object
tracking into a single, frame-by-frame processing loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .ocr.detector import OCRDetector, OCRResult
from .keypoints.detector import KeypointDetector, KeypointResult
from .tracking.tracker import ObjectTracker, TrackResult


@dataclass
class PipelineResult:
    """Aggregated results for one processed frame."""

    frame_index: int
    ocr_results: List[OCRResult] = field(default_factory=list)
    keypoint_result: Optional[KeypointResult] = None
    track_results: List[TrackResult] = field(default_factory=list)


class Pipeline:
    """Run OCR, keypoint detection, and object tracking on every frame.

    Parameters
    ----------
    ocr_detector:
        Pre-configured :class:`~src.ocr.detector.OCRDetector` instance.
        Pass ``None`` to skip OCR.
    keypoint_detector:
        Pre-configured :class:`~src.keypoints.detector.KeypointDetector`
        instance.  Pass ``None`` to skip keypoint detection.
    object_tracker:
        Pre-configured :class:`~src.tracking.tracker.ObjectTracker`
        instance.  Pass ``None`` to skip tracking.
    draw_results:
        When ``True`` each processed frame will be annotated with the
        detection / tracking overlays and returned in the result (stored
        under ``annotated_frame``).  Default ``False``.
    """

    def __init__(
        self,
        ocr_detector: Optional[OCRDetector] = None,
        keypoint_detector: Optional[KeypointDetector] = None,
        object_tracker: Optional[ObjectTracker] = None,
        draw_results: bool = False,
    ) -> None:
        self.ocr = ocr_detector
        self.kp = keypoint_detector
        self.tracker = object_tracker
        self.draw_results = draw_results
        self._frame_index: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> PipelineResult:
        """Process a single *frame* through all enabled sub-modules.

        Parameters
        ----------
        frame:
            BGR image as a NumPy array.

        Returns
        -------
        PipelineResult
        """
        result = PipelineResult(frame_index=self._frame_index)

        if self.ocr is not None:
            result.ocr_results = self.ocr.detect(frame)

        if self.kp is not None:
            result.keypoint_result = self.kp.detect(frame)

        if self.tracker is not None:
            result.track_results = self.tracker.update(frame)

        if self.draw_results:
            self._annotate(frame, result)

        self._frame_index += 1
        return result

    def reset(self) -> None:
        """Reset frame counter and clear all active trackers."""
        self._frame_index = 0
        if self.tracker is not None:
            self.tracker.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _annotate(self, frame: np.ndarray, result: PipelineResult) -> None:
        if self.ocr is not None:
            self.ocr.draw(frame, result.ocr_results)
        if self.kp is not None and result.keypoint_result is not None:
            self.kp.draw(frame, result.keypoint_result)
        if self.tracker is not None:
            self.tracker.draw(frame, result.track_results)
