package com.example.mugunghwa.tracking

import kotlin.math.hypot

class MovementScorer(
    defaultThreshold: Float = 0.006f
) {
    var movementThreshold: Float = defaultThreshold
        private set

    private val previousLandmarksByTrack = mutableMapOf<Int, Map<Int, LandmarkPoint>>()
    private val redBaselineLandmarksByTrack = mutableMapOf<Int, Map<Int, LandmarkPoint>>()
    private val redBaselineBboxByTrack = mutableMapOf<Int, android.graphics.RectF>()

    fun frameDeltaScore(track: PlayerTrack): Float {
        val current = track.landmarks
            .reliableKeyLandmarks()
        val previous = previousLandmarksByTrack[track.id]
        val score = scoreAgainstReference(track, current, previous)
        previousLandmarksByTrack[track.id] = current
        return score
    }

    fun redLightScore(track: PlayerTrack): Float {
        val current = track.landmarks
            .reliableKeyLandmarks()
        val baseline = redBaselineLandmarksByTrack[track.id] ?: current.also {
            redBaselineLandmarksByTrack[track.id] = it
        }
        previousLandmarksByTrack[track.id] = current
        val landmarkScore = scoreAgainstReference(track, current, baseline)
        val bboxScore = bboxScoreAgainstRedBaseline(track)
        val score = maxOf(landmarkScore, bboxScore)
        track.movementHistory.removeLastOrNull()
        track.movementHistory += score
        return score
    }

    private fun scoreAgainstReference(
        track: PlayerTrack,
        current: Map<Int, LandmarkPoint>,
        reference: Map<Int, LandmarkPoint>?
    ): Float {
        val raw = if (reference == null) {
            0f
        } else {
            val distances = current.mapNotNull { (index, point) ->
                reference[index]?.let { old -> hypot(point.x - old.x, point.y - old.y) }
            }
            if (distances.isEmpty()) 0f else distances.average().toFloat()
        }

        track.movementHistory += raw
        if (track.movementHistory.size > 120) {
            track.movementHistory.removeAt(0)
        }
        return raw
    }

    fun updateGreenBaseline(track: PlayerTrack) {
        previousLandmarksByTrack[track.id] = track.landmarks.reliableKeyLandmarks()
        redBaselineLandmarksByTrack.remove(track.id)
        redBaselineBboxByTrack.remove(track.id)
        track.redViolationFrames = 0
    }

    fun captureRedLightBaseline(track: PlayerTrack) {
        val baseline = track.landmarks.reliableKeyLandmarks()
        redBaselineBboxByTrack[track.id] = android.graphics.RectF(track.bbox)
        if (baseline.isNotEmpty()) {
            redBaselineLandmarksByTrack[track.id] = baseline
            previousLandmarksByTrack[track.id] = baseline
            track.redViolationFrames = 0
        }
    }

    fun setCalibratedThreshold(value: Float) {
        movementThreshold = value.coerceIn(0.006f, 0.035f)
    }

    fun reset() {
        previousLandmarksByTrack.clear()
        redBaselineLandmarksByTrack.clear()
        redBaselineBboxByTrack.clear()
        movementThreshold = 0.006f
    }

    private fun bboxScoreAgainstRedBaseline(track: PlayerTrack): Float {
        val baseline = redBaselineBboxByTrack[track.id] ?: return 0f
        val centerDx = track.bbox.centerX() - baseline.centerX()
        val centerDy = track.bbox.centerY() - baseline.centerY()
        val centerMove = hypot(centerDx, centerDy)
        val sizeChange = kotlin.math.abs(track.bbox.width() - baseline.width()) +
            kotlin.math.abs(track.bbox.height() - baseline.height())
        return centerMove + sizeChange * 0.15f
    }

    private fun List<LandmarkPoint>.reliableKeyLandmarks(): Map<Int, LandmarkPoint> {
        return filter {
            it.index in KEY_LANDMARKS &&
                it.visibility >= MIN_CONFIDENCE &&
                it.presence >= MIN_CONFIDENCE
        }.associateBy { it.index }
    }

    companion object {
        private const val MIN_CONFIDENCE = 0.25f
        val KEY_LANDMARKS = setOf(0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
    }
}
