"""
Object tracker module.

Wraps OpenCV's built-in tracker implementations so they can be used
with a simple, uniform API.  Supports multi-object tracking through a
lightweight registry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class TrackResult:
    """Tracking result for one object in one frame."""

    track_id: int
    bbox: Tuple[int, int, int, int]   # (x, y, w, h)
    success: bool


class ObjectTracker:
    """Track one or more objects across video frames.

    Parameters
    ----------
    tracker_type:
        OpenCV tracker algorithm to use.  Supported values:
        ``'csrt'`` (default), ``'kcf'``, ``'mosse'``.
    """

    _FACTORIES: Dict[str, str] = {
        "csrt": "TrackerCSRT",
        "kcf": "TrackerKCF",
        "mosse": "legacy_TrackerMOSSE",
    }

    def __init__(self, tracker_type: str = "csrt") -> None:
        tracker_type = tracker_type.lower()
        if tracker_type not in self._FACTORIES:
            raise ValueError(
                f"Unsupported tracker '{tracker_type}'. "
                f"Choose one of {list(self._FACTORIES)}."
            )
        self.tracker_type = tracker_type
        self._trackers: Dict[int, cv2.Tracker] = {}
        self._next_id: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def init(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> int:
        """Register a new object to track.

        Parameters
        ----------
        frame:
            The current video frame (BGR).
        bbox:
            Initial bounding box as ``(x, y, width, height)``.

        Returns
        -------
        int
            Unique track ID assigned to this object.
        """
        tracker = self._create_tracker()
        tracker.init(frame, bbox)
        track_id = self._next_id
        self._trackers[track_id] = tracker
        self._next_id += 1
        return track_id

    def update(self, frame: np.ndarray) -> List[TrackResult]:
        """Update all active trackers with the next *frame*.

        Parameters
        ----------
        frame:
            The new video frame (BGR).

        Returns
        -------
        list[TrackResult]
            One result per active tracked object.
        """
        results: List[TrackResult] = []
        for track_id, tracker in list(self._trackers.items()):
            success, bbox_raw = tracker.update(frame)
            bbox = tuple(int(v) for v in bbox_raw)  # type: ignore[arg-type]
            results.append(TrackResult(track_id=track_id, bbox=bbox, success=success))
        return results

    def remove(self, track_id: int) -> None:
        """Remove the tracker with *track_id* from the registry."""
        self._trackers.pop(track_id, None)

    def clear(self) -> None:
        """Remove all active trackers."""
        self._trackers.clear()

    @property
    def active_ids(self) -> List[int]:
        """Return all currently active track IDs."""
        return list(self._trackers.keys())

    def draw(self, frame: np.ndarray, results: List[TrackResult]) -> np.ndarray:
        """Draw tracking bounding boxes onto *frame*.

        Parameters
        ----------
        frame:
            Source image (modified in-place).
        results:
            Tracking results from :meth:`update`.

        Returns
        -------
        np.ndarray
            Annotated image.
        """
        for r in results:
            color = (0, 255, 255) if r.success else (0, 0, 255)
            x, y, w, h = r.bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame, f"ID {r.track_id}", (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            )
        return frame

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _create_tracker(self) -> cv2.Tracker:
        factory_name = self._FACTORIES[self.tracker_type]
        # OpenCV 4.5+: trackers live directly under cv2
        # Some builds expose them under cv2.legacy
        if hasattr(cv2, factory_name):
            return getattr(cv2, factory_name).create()
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, factory_name):
            return getattr(cv2.legacy, factory_name).create()
        raise RuntimeError(
            f"Could not find tracker '{factory_name}' in your OpenCV build. "
            "Install opencv-contrib-python."
        )
