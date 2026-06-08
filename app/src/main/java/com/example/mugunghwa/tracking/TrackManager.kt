package com.example.mugunghwa.tracking

import android.graphics.PointF
import android.graphics.RectF
import kotlin.math.hypot

class TrackManager(
    private val matchThreshold: Float = 0.32f,
    private val maxMissedFrames: Int = 12,
    private val maxVisibleMissedFrames: Int = 3
) {
    private val tracks = mutableListOf<PlayerTrack>()
    private var nextTrackId = 1

    fun update(detections: List<PlayerPose>, timestampMs: Long): List<PlayerTrack> {
        val unmatchedDetectionIndexes = detections.indices.toMutableSet()
        val matchedTrackIds = mutableSetOf<Int>()

        tracks.filter { it.active }.forEach { track ->
            var bestIndex: Int? = null
            var bestScore = Float.MAX_VALUE
            unmatchedDetectionIndexes.forEach { index ->
                val score = matchScore(track, detections[index])
                if (score < bestScore) {
                    bestScore = score
                    bestIndex = index
                }
            }

            if (bestIndex != null && bestScore <= matchThreshold) {
                val pose = detections[bestIndex!!]
                val incomingCenter = pose.bbox.center()
                val oldCenter = track.lastCenter
                track.bbox = stabilizeBbox(track.bbox, pose.bbox)
                track.landmarks = pose.landmarks
                track.imageWidth = pose.imageWidth
                track.imageHeight = pose.imageHeight
                track.velocity = PointF(
                    lerp(track.velocity.x, incomingCenter.x - oldCenter.x, 0.45f),
                    lerp(track.velocity.y, incomingCenter.y - oldCenter.y, 0.45f)
                )
                track.lastCenter = track.bbox.center()
                track.detectionHits += 1
                if (track.detectionHits >= 2) {
                    track.confirmed = true
                }
                track.missedFrames = 0
                track.lastUpdatedMs = timestampMs
                track.active = true
                unmatchedDetectionIndexes.remove(bestIndex!!)
                matchedTrackIds.add(track.id)
            }
        }

        tracks.filter { it.active && it.id !in matchedTrackIds }.forEach { track ->
            track.missedFrames += 1
            if (track.missedFrames > maxMissedFrames) {
                track.active = false
            }
        }

        unmatchedDetectionIndexes.forEach { index ->
            val pose = detections[index]
            tracks += PlayerTrack(
                id = nextTrackId++,
                bbox = RectF(pose.bbox),
                landmarks = pose.landmarks,
                imageWidth = pose.imageWidth,
                imageHeight = pose.imageHeight,
                lastCenter = pose.bbox.center(),
                velocity = PointF(),
                lastUpdatedMs = timestampMs
            )
        }

        return activeTracks()
    }

    fun activeTracks(): List<PlayerTrack> {
        return tracks.filter { it.active && it.confirmed && it.missedFrames <= maxVisibleMissedFrames }
    }

    fun totalTrackedCount(): Int = tracks.size

    fun reset() {
        tracks.clear()
        nextTrackId = 1
    }
}

private fun RectF.center(): PointF = PointF(centerX(), centerY())

private fun PointF.distanceTo(other: PointF): Float = hypot(x - other.x, y - other.y)

private fun matchScore(track: PlayerTrack, detection: PlayerPose): Float {
    val predictedCenter = PointF(
        track.lastCenter.x + track.velocity.x,
        track.lastCenter.y + track.velocity.y
    )
    val centerDistance = predictedCenter.distanceTo(detection.bbox.center())
    val overlap = iou(track.bbox, detection.bbox)
    val areaRatio = areaRatio(track.bbox, detection.bbox)
    val areaPenalty = kotlin.math.abs(1f - areaRatio).coerceAtMost(1f) * 0.08f
    return centerDistance * 0.72f + (1f - overlap) * 0.20f + areaPenalty
}

private fun stabilizeBbox(previous: RectF, incoming: RectF): RectF {
    val previousArea = previous.area().coerceAtLeast(0.0001f)
    val incomingArea = incoming.area().coerceAtLeast(0.0001f)
    val areaRatio = incomingArea / previousArea
    val centerJump = previous.center().distanceTo(incoming.center())
    val suspicious = areaRatio > 1.7f || areaRatio < 0.55f || centerJump > 0.12f
    val alpha = if (suspicious) 0.55f else 0.90f
    return RectF(
        lerp(previous.left, incoming.left, alpha),
        lerp(previous.top, incoming.top, alpha),
        lerp(previous.right, incoming.right, alpha),
        lerp(previous.bottom, incoming.bottom, alpha)
    )
}

private fun RectF.area(): Float = width().coerceAtLeast(0f) * height().coerceAtLeast(0f)

private fun lerp(start: Float, end: Float, alpha: Float): Float = start + (end - start) * alpha

private fun iou(a: RectF, b: RectF): Float {
    val left = maxOf(a.left, b.left)
    val top = maxOf(a.top, b.top)
    val right = minOf(a.right, b.right)
    val bottom = minOf(a.bottom, b.bottom)
    val intersection = ((right - left).coerceAtLeast(0f)) * ((bottom - top).coerceAtLeast(0f))
    val union = a.area() + b.area() - intersection
    if (union <= 0f) return 0f
    return intersection / union
}

private fun areaRatio(a: RectF, b: RectF): Float {
    val small = minOf(a.area(), b.area()).coerceAtLeast(0.0001f)
    val large = maxOf(a.area(), b.area()).coerceAtLeast(0.0001f)
    return small / large
}
