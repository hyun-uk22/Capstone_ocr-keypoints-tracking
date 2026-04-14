"""
OCR detector module.

Uses EasyOCR to detect and recognise text regions in a frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np


@dataclass
class OCRResult:
    """Holds a single text detection result."""

    bbox: List[List[int]]   # four corner points [[x,y], ...]
    text: str
    confidence: float


class OCRDetector:
    """Detect and recognise text in images using EasyOCR.

    Parameters
    ----------
    languages:
        List of language codes supported by EasyOCR (default ``['en']``).
    gpu:
        Whether to use a GPU for inference (default ``False``).
    min_confidence:
        Minimum confidence threshold; results below this value are
        discarded (default ``0.5``).
    """

    def __init__(
        self,
        languages: List[str] | None = None,
        gpu: bool = False,
        min_confidence: float = 0.5,
    ) -> None:
        if languages is None:
            languages = ["en"]
        self.languages = languages
        self.gpu = gpu
        self.min_confidence = min_confidence
        self._reader = None  # lazy-load to avoid slow import at module level

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[OCRResult]:
        """Run OCR on *frame* and return a list of :class:`OCRResult`.

        Parameters
        ----------
        frame:
            A BGR or grayscale image as a NumPy array.

        Returns
        -------
        list[OCRResult]
            Detected text regions filtered by ``min_confidence``.
        """
        if frame is None or frame.size == 0:
            return []

        reader = self._get_reader()
        # EasyOCR expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.ndim == 3 else frame
        raw = reader.readtext(rgb)

        results: List[OCRResult] = []
        for bbox, text, conf in raw:
            if conf >= self.min_confidence:
                results.append(OCRResult(bbox=bbox, text=text, confidence=conf))
        return results

    def draw(self, frame: np.ndarray, results: List[OCRResult]) -> np.ndarray:
        """Draw bounding boxes and recognised text onto *frame*.

        Parameters
        ----------
        frame:
            Source image (modified in-place).
        results:
            OCR results returned by :meth:`detect`.

        Returns
        -------
        np.ndarray
            The annotated image.
        """
        for r in results:
            pts = np.array(r.bbox, dtype=np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            origin = (pts[0][0], pts[0][1] - 5)
            label = f"{r.text} ({r.confidence:.2f})"
            cv2.putText(
                frame, label, origin,
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )
        return frame

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_reader(self):
        """Lazy-load EasyOCR reader on first use."""
        if self._reader is None:
            import easyocr  # noqa: PLC0415 – intentional lazy import
            self._reader = easyocr.Reader(self.languages, gpu=self.gpu)
        return self._reader
