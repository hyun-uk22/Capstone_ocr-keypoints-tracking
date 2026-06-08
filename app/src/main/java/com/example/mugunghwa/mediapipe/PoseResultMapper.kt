package com.example.mugunghwa.mediapipe

import android.graphics.RectF
import com.example.mugunghwa.tracking.LandmarkPoint
import com.example.mugunghwa.tracking.PlayerPose
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import kotlin.math.abs

object PoseResultMapper {
    private val primaryBboxLandmarks = setOf(0, 11, 12, 23, 24, 25, 26, 27, 28)
    private val humanGateLandmarks = setOf(0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
    private val limbLandmarks = setOf(13, 14, 15, 16, 25, 26, 27, 28)

    fun map(result: PoseLandmarkerResult, imageWidth: Int, imageHeight: Int): List<PlayerPose> {
        return result.landmarks().mapNotNull { poseLandmarks ->
            val points = poseLandmarks.mapIndexed { index, landmark ->
                LandmarkPoint(
                    index = index,
                    x = landmark.x(),
                    y = landmark.y(),
                    z = landmark.z(),
                    visibility = optionalValue(landmark.visibility()),
                    presence = optionalValue(landmark.presence())
                )
            }
            if (!looksLikeHumanPose(points)) return@mapNotNull null
            val bbox = boundingBox(poseLandmarks) ?: return@mapNotNull null
            PlayerPose(
                bbox = bbox,
                landmarks = points,
                imageWidth = imageWidth,
                imageHeight = imageHeight
            )
        }.sortedByDescending { it.bbox.width() * it.bbox.height() }
            .take(MAX_PLAYERS)
    }

    private fun boundingBox(landmarks: List<NormalizedLandmark>): RectF? {
        val reliable = reliableLandmarks(landmarks, primaryOnly = false)
        if (reliable.size < 4) return null
        val byIndex = reliableLandmarksByIndex(landmarks)
        val torsoBox = torsoAnchoredBox(byIndex)
        val candidateBox = if (torsoBox != null) {
            expandTorsoBoxWithNearbyBodyPoints(torsoBox, reliable)
        } else {
            minMaxBox(reliable)
        }

        val minX = candidateBox.left.coerceIn(0f, 1f)
        val minY = candidateBox.top.coerceIn(0f, 1f)
        val maxX = candidateBox.right.coerceIn(0f, 1f)
        val maxY = candidateBox.bottom.coerceIn(0f, 1f)
        val rawWidth = maxX - minX
        val rawHeight = maxY - minY
        if (!isPlausibleHumanBox(rawWidth, rawHeight)) return null

        val padX = ((maxX - minX) * 0.35f).coerceAtLeast(0.04f)
        val padY = ((maxY - minY) * 0.35f).coerceAtLeast(0.04f)
        return RectF(
            (minX - padX).coerceIn(0f, 1f),
            (minY - padY).coerceIn(0f, 1f),
            (maxX + padX).coerceIn(0f, 1f),
            (maxY + padY).coerceIn(0f, 1f)
        )
    }

    private fun torsoAnchoredBox(points: Map<Int, NormalizedLandmark>): RectF? {
        val shoulders = listOfNotNull(points[11], points[12])
        val hips = listOfNotNull(points[23], points[24])
        if (shoulders.isEmpty() && hips.isEmpty()) return null

        val torsoPoints = shoulders + hips
        val centerX = torsoPoints.map { it.x() }.average().toFloat()
        val shoulderY = shoulders.map { it.y() }.averageOrNull()
        val hipY = hips.map { it.y() }.averageOrNull()
        val topY = shoulderY ?: torsoPoints.minOf { it.y() }
        val bottomY = hipY ?: torsoPoints.maxOf { it.y() }
        val torsoHeight = (bottomY - topY).coerceAtLeast(0.05f)

        val shoulderWidth = pairWidth(points[11], points[12])
        val hipWidth = pairWidth(points[23], points[24])
        val torsoWidth = maxOf(shoulderWidth, hipWidth, torsoHeight * 0.35f).coerceAtMost(torsoHeight * 1.2f)

        val width = (torsoWidth * 1.9f).coerceIn(0.055f, 0.38f)
        val height = (torsoHeight * 2.45f).coerceIn(0.11f, 0.72f)
        val top = topY - height * 0.18f
        return RectF(
            centerX - width / 2f,
            top,
            centerX + width / 2f,
            top + height
        )
    }

    private fun expandTorsoBoxWithNearbyBodyPoints(torsoBox: RectF, points: List<NormalizedLandmark>): RectF {
        val maxLeft = torsoBox.left - torsoBox.width() * 0.22f
        val maxTop = torsoBox.top - torsoBox.height() * 0.12f
        val maxRight = torsoBox.right + torsoBox.width() * 0.22f
        val maxBottom = torsoBox.bottom + torsoBox.height() * 0.16f
        val nearby = points.filter {
            it.x() in maxLeft..maxRight && it.y() in maxTop..maxBottom
        }
        if (nearby.isEmpty()) return torsoBox

        val minMax = minMaxBox(nearby)
        return RectF(
            minOf(torsoBox.left, minMax.left).coerceAtLeast(maxLeft),
            minOf(torsoBox.top, minMax.top).coerceAtLeast(maxTop),
            maxOf(torsoBox.right, minMax.right).coerceAtMost(maxRight),
            maxOf(torsoBox.bottom, minMax.bottom).coerceAtMost(maxBottom)
        )
    }

    private fun minMaxBox(points: List<NormalizedLandmark>): RectF {
        return RectF(
            points.minOf { it.x() },
            points.minOf { it.y() },
            points.maxOf { it.x() },
            points.maxOf { it.y() }
        )
    }

    private fun reliableLandmarks(
        landmarks: List<NormalizedLandmark>,
        primaryOnly: Boolean
    ): List<NormalizedLandmark> {
        return landmarks.mapIndexedNotNull { index, landmark ->
            if (primaryOnly && index !in primaryBboxLandmarks) return@mapIndexedNotNull null
            if (!primaryOnly && index !in humanGateLandmarks) return@mapIndexedNotNull null
            val visibility = optionalValue(landmark.visibility())
            val presence = optionalValue(landmark.presence())
            if (visibility < 0.20f || presence < 0.20f) return@mapIndexedNotNull null
            landmark
        }
    }

    private fun reliableLandmarksByIndex(
        landmarks: List<NormalizedLandmark>
    ): Map<Int, NormalizedLandmark> {
        return landmarks.mapIndexedNotNull { index, landmark ->
            if (index !in humanGateLandmarks) return@mapIndexedNotNull null
            val visibility = optionalValue(landmark.visibility())
            val presence = optionalValue(landmark.presence())
            if (visibility < 0.20f || presence < 0.20f) return@mapIndexedNotNull null
            index to landmark
        }.toMap()
    }

    private fun looksLikeHumanPose(points: List<LandmarkPoint>): Boolean {
        val reliable = points.filter {
            it.index in humanGateLandmarks && it.visibility >= 0.20f && it.presence >= 0.20f
        }
        if (reliable.size < 7) return false

        val byIndex = reliable.associateBy { it.index }
        val hasShoulderPair = byIndex[11] != null && byIndex[12] != null
        val hasHipPair = byIndex[23] != null && byIndex[24] != null

        val hasUpperBody = byIndex[11] != null || byIndex[12] != null || byIndex[0] != null
        val hasLowerBody = byIndex[23] != null || byIndex[24] != null || byIndex[25] != null ||
            byIndex[26] != null || byIndex[27] != null || byIndex[28] != null
        if (!hasUpperBody || !hasLowerBody) return false

        val limbCount = reliable.count { it.index in limbLandmarks }
        if (limbCount < 2) return false

        val bodyHeight = (reliable.maxOf { it.y } - reliable.minOf { it.y }).coerceAtLeast(0.001f)
        val bodyWidth = reliable.maxOf { it.x } - reliable.minOf { it.x }

        if (bodyHeight < 0.045f || bodyWidth < 0.018f) return false

        val hasPlausibleShoulders = hasShoulderPair &&
            isPlausibleHorizontalPair(byIndex[11]!!, byIndex[12]!!, bodyHeight, minWidthRatio = 0.09f)
        val hasPlausibleHips = hasHipPair &&
            isPlausibleHorizontalPair(byIndex[23]!!, byIndex[24]!!, bodyHeight, minWidthRatio = 0.07f)
        if (!hasPlausibleShoulders && !hasPlausibleHips) return false

        if (!hasPlausibleTorsoOrder(byIndex)) return false

        return when {
            hasPlausibleShoulders && hasPlausibleHips -> limbCount >= 2
            reliable.size >= 8 -> limbCount >= 3
            else -> false
        }
    }

    private fun isPlausibleHumanBox(width: Float, height: Float): Boolean {
        if (width <= 0.018f || height <= 0.045f) return false
        val aspect = width / height
        return aspect in 0.24f..1.45f
    }

    private fun isPlausibleHorizontalPair(
        left: LandmarkPoint,
        right: LandmarkPoint,
        bodyHeight: Float,
        minWidthRatio: Float
    ): Boolean {
        val pairWidth = abs(left.x - right.x)
        val pairTilt = abs(left.y - right.y)
        return pairWidth / bodyHeight >= minWidthRatio && pairTilt / bodyHeight <= 0.35f
    }

    private fun hasPlausibleTorsoOrder(points: Map<Int, LandmarkPoint>): Boolean {
        val leftShoulder = points[11]
        val rightShoulder = points[12]
        val leftHip = points[23]
        val rightHip = points[24]
        val nose = points[0]

        val shoulderY = listOfNotNull(leftShoulder?.y, rightShoulder?.y).averageOrNull()
        val hipY = listOfNotNull(leftHip?.y, rightHip?.y).averageOrNull()
        if (shoulderY != null && hipY != null && hipY <= shoulderY + 0.015f) return false
        if (nose != null && shoulderY != null && nose.y >= shoulderY + 0.12f) return false
        return true
    }

    private fun List<Float>.averageOrNull(): Float? {
        if (isEmpty()) return null
        return average().toFloat()
    }

    private fun pairWidth(left: NormalizedLandmark?, right: NormalizedLandmark?): Float {
        if (left == null || right == null) return 0f
        return abs(left.x() - right.x())
    }

    private fun optionalValue(value: java.util.Optional<Float>): Float = value.orElse(1f)

    private const val MAX_PLAYERS = 5
}
