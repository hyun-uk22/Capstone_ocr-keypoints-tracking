package com.example.mugunghwa.tracking

import android.graphics.PointF
import android.graphics.RectF

data class LandmarkPoint(
    val index: Int,
    val x: Float,
    val y: Float,
    val z: Float,
    val visibility: Float,
    val presence: Float
)

data class PlayerPose(
    val bbox: RectF,
    val landmarks: List<LandmarkPoint>,
    val imageWidth: Int,
    val imageHeight: Int
)

data class PlayerTrack(
    val id: Int,
    var label: String? = null,
    var bbox: RectF = RectF(),
    var landmarks: List<LandmarkPoint> = emptyList(),
    var imageWidth: Int = 0,
    var imageHeight: Int = 0,
    var lastCenter: PointF = PointF(),
    var velocity: PointF = PointF(),
    val movementHistory: MutableList<Float> = mutableListOf(),
    var redViolationFrames: Int = 0,
    var eliminated: Boolean = false,
    var missedFrames: Int = 0,
    var lastUpdatedMs: Long = 0L,
    val ocrCandidates: MutableMap<String, Int> = mutableMapOf(),
    var active: Boolean = true,
    var detectionHits: Int = 1,
    var confirmed: Boolean = false,
    var overlapping: Boolean = false,
    var motionHoldFrames: Int = 0
) {
    val displayName: String
        get() = label ?: "#$id"

    val latestMovementScore: Float
        get() = movementHistory.lastOrNull() ?: 0f
}
