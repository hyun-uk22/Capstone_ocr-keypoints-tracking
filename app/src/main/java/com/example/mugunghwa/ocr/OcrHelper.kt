package com.example.mugunghwa.ocr

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import com.example.mugunghwa.game.GameEngine
import com.example.mugunghwa.tracking.LandmarkPoint
import com.example.mugunghwa.tracking.PlayerTrack
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.korean.KoreanTextRecognizerOptions
import java.util.concurrent.atomic.AtomicBoolean

class OcrHelper(
    private val gameEngine: GameEngine,
    private val minIntervalMs: Long = 1000L
) {
    private val recognizer = TextRecognition.getClient(KoreanTextRecognizerOptions.Builder().build())
    private val running = AtomicBoolean(false)
    private var lastRunMs = 0L
    private var nextTargetOffset = 0
    private val candidateRegex = Regex("""[A-Z]?[0-9OQILSZB]{1,3}""", RegexOption.IGNORE_CASE)

    fun maybeRun(bitmap: Bitmap, tracks: List<PlayerTrack>, nowMs: Long) {
        if (nowMs - lastRunMs < minIntervalMs) {
            bitmap.recycleIfNeeded()
            return
        }
        if (!running.compareAndSet(false, true)) {
            bitmap.recycleIfNeeded()
            return
        }
        lastRunMs = nowMs

        val targets = tracks.filter { isOcrSafeTarget(it, tracks) }
        val target = chooseTarget(targets)
        if (target == null) {
            bitmap.recycleIfNeeded()
            running.set(false)
            return
        }

        val roi = chestRoi(target) ?: fallbackRoi(target)
        val crop = cropAndEnhance(bitmap, roi)
        bitmap.recycleIfNeeded()
        if (crop == null) {
            running.set(false)
            return
        }

        recognizer.process(InputImage.fromBitmap(crop, 0))
            .addOnSuccessListener { result ->
                extractCandidates(result.text).forEach { candidate ->
                    if (gameEngine.recordOcrCandidate(target.id, candidate)) {
                        gameEngine.lockOcrLabel(target.id, candidate)
                        return@forEach
                    }
                }
            }
            .addOnCompleteListener {
                crop.recycle()
                running.set(false)
            }
    }

    fun close() {
        recognizer.close()
    }

    private fun chooseTarget(targets: List<PlayerTrack>): PlayerTrack? {
        if (targets.isEmpty()) return null
        val target = targets[nextTargetOffset % targets.size]
        nextTargetOffset = (nextTargetOffset + 1) % targets.size
        return target
    }

    private fun isOcrSafeTarget(track: PlayerTrack, tracks: List<PlayerTrack>): Boolean {
        if (!track.active || track.eliminated || track.label != null || track.overlapping) return false
        if (track.missedFrames > 0) return false

        val roi = chestRoi(track) ?: fallbackRoi(track)
        return tracks.none { other ->
            other.id != track.id &&
                other.active &&
                !other.eliminated &&
                (iou(track.bbox, other.bbox) > 0.12f || intersectionRatio(roi, other.bbox) > 0.08f)
        }
    }

    private fun chestRoi(track: PlayerTrack): RectF? {
        val points = track.landmarks.associateBy { it.index }
        val leftShoulder = points[11]?.reliable() ?: return null
        val rightShoulder = points[12]?.reliable() ?: return null
        val leftHip = points[23]?.reliable()
        val rightHip = points[24]?.reliable()

        val shoulderCenterX = (leftShoulder.x + rightShoulder.x) / 2f
        val shoulderY = (leftShoulder.y + rightShoulder.y) / 2f
        val shoulderWidth = kotlin.math.abs(leftShoulder.x - rightShoulder.x)
        val hipY = listOfNotNull(leftHip?.y, rightHip?.y).averageOrNull()
        val torsoHeight = ((hipY ?: track.bbox.bottom) - shoulderY).coerceAtLeast(track.bbox.height() * 0.35f)

        val roiWidth = (shoulderWidth * 1.9f).coerceAtLeast(track.bbox.width() * 0.45f)
        val top = shoulderY + torsoHeight * 0.10f
        val bottom = shoulderY + torsoHeight * 0.72f
        return RectF(
            shoulderCenterX - roiWidth / 2f,
            top,
            shoulderCenterX + roiWidth / 2f,
            bottom
        ).clamped()
    }

    private fun fallbackRoi(track: PlayerTrack): RectF {
        val box = track.bbox
        return RectF(
            box.left + box.width() * 0.18f,
            box.top + box.height() * 0.18f,
            box.right - box.width() * 0.18f,
            box.top + box.height() * 0.62f
        ).clamped()
    }

    private fun cropAndEnhance(bitmap: Bitmap, normalizedBox: RectF): Bitmap? {
        val rect = Rect(
            (normalizedBox.left * bitmap.width).toInt().coerceIn(0, bitmap.width - 1),
            (normalizedBox.top * bitmap.height).toInt().coerceIn(0, bitmap.height - 1),
            (normalizedBox.right * bitmap.width).toInt().coerceIn(1, bitmap.width),
            (normalizedBox.bottom * bitmap.height).toInt().coerceIn(1, bitmap.height)
        )
        if (rect.width() <= 10 || rect.height() <= 10) return null

        val crop = Bitmap.createBitmap(bitmap, rect.left, rect.top, rect.width(), rect.height())
        val scaled = Bitmap.createScaledBitmap(crop, crop.width * 3, crop.height * 3, true)
        crop.recycle()

        val enhanced = Bitmap.createBitmap(scaled.width, scaled.height, Bitmap.Config.ARGB_8888)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            colorFilter = ColorMatrixColorFilter(
                ColorMatrix(
                    floatArrayOf(
                        1.45f, 0f, 0f, 0f, 10f,
                        0f, 1.45f, 0f, 0f, 10f,
                        0f, 0f, 1.45f, 0f, 10f,
                        0f, 0f, 0f, 1f, 0f
                    )
                )
            )
        }
        Canvas(enhanced).drawBitmap(scaled, 0f, 0f, paint)
        scaled.recycle()
        return enhanced
    }

    private fun extractCandidates(text: String): List<String> {
        val compact = text.uppercase()
            .replace(" ", "")
            .replace("-", "")
            .replace("_", "")

        return candidateRegex.findAll(compact)
            .mapNotNull { normalizeCandidate(it.value) }
            .distinct()
            .take(3)
            .toList()
    }

    private fun normalizeCandidate(raw: String): String? {
        val normalized = raw.uppercase()
            .replace('O', '0')
            .replace('Q', '0')
            .replace('I', '1')
            .replace('L', '1')
            .replace('S', '5')
            .replace('Z', '2')
            .replace('B', '8')
        val digits = normalized.filter { it.isDigit() }
        if (digits.isEmpty() || digits.length > 3) return null
        return digits.padStart(2, '0')
    }

    private fun LandmarkPoint.reliable(): LandmarkPoint? {
        return takeIf { visibility >= 0.20f && presence >= 0.20f }
    }

    private fun RectF.clamped(): RectF {
        return RectF(
            left.coerceIn(0f, 1f),
            top.coerceIn(0f, 1f),
            right.coerceIn(0f, 1f),
            bottom.coerceIn(0f, 1f)
        )
    }

    private fun List<Float>.averageOrNull(): Float? {
        if (isEmpty()) return null
        return average().toFloat()
    }

    private fun iou(a: RectF, b: RectF): Float {
        val left = maxOf(a.left, b.left)
        val top = maxOf(a.top, b.top)
        val right = minOf(a.right, b.right)
        val bottom = minOf(a.bottom, b.bottom)
        val intersection = (right - left).coerceAtLeast(0f) * (bottom - top).coerceAtLeast(0f)
        val union = a.width() * a.height() + b.width() * b.height() - intersection
        if (union <= 0f) return 0f
        return intersection / union
    }

    private fun intersectionRatio(target: RectF, other: RectF): Float {
        val left = maxOf(target.left, other.left)
        val top = maxOf(target.top, other.top)
        val right = minOf(target.right, other.right)
        val bottom = minOf(target.bottom, other.bottom)
        val intersection = (right - left).coerceAtLeast(0f) * (bottom - top).coerceAtLeast(0f)
        val targetArea = target.width() * target.height()
        if (targetArea <= 0f) return 0f
        return intersection / targetArea
    }

    private fun Bitmap.recycleIfNeeded() {
        if (!isRecycled) recycle()
    }
}
