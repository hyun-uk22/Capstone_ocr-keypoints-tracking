package com.example.mugunghwa.yolo

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import com.example.mugunghwa.pose.PoseEstimator
import com.example.mugunghwa.tracking.LandmarkPoint
import com.example.mugunghwa.tracking.PlayerPose
import java.io.IOException
import java.nio.FloatBuffer
import java.util.concurrent.atomic.AtomicBoolean

data class YOLOPoseConfig(
    val modelAssetPath: String = "yolo26n-pose.onnx",
    val maxPoses: Int = 5,
    val confidenceThreshold: Float = 0.25f,
    val iouThreshold: Float = 0.45f,
    val keypointScoreThreshold: Float = 0.10f
)

class YOLOPoseHelper(
    private val context: Context,
    private val config: YOLOPoseConfig = YOLOPoseConfig(),
    private val onResult: (List<PlayerPose>, Long) -> Unit,
    private val onError: (String) -> Unit
) : PoseEstimator {
    private val sessionLock = Any()
    private val busy = AtomicBoolean(false)
    private var environment: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var ready = false
    private var inputName = "images"
    private var inputSize = 640

    override fun setup() {
        synchronized(sessionLock) {
            if (!assetExists(config.modelAssetPath)) {
                ready = false
                onError("${config.modelAssetPath} is missing. Place the YOLO26 pose ONNX model in app/src/main/assets.")
                return
            }

            try {
                closeLocked()
                val env = OrtEnvironment.getEnvironment()
                val options = OrtSession.SessionOptions().apply {
                    setIntraOpNumThreads(4)
                }
                val modelBytes = context.assets.open(config.modelAssetPath).use { it.readBytes() }
                val ortSession = env.createSession(modelBytes, options)
                environment = env
                session = ortSession
                configureInput(ortSession)
                ready = true
                onError("")
            } catch (e: Exception) {
                closeLocked()
                onError(e.message ?: "YOLO26 pose initialization failed")
            }
        }
    }

    override fun detectLiveStream(bitmap: Bitmap, rotationDegrees: Int, timestampMs: Long): Boolean {
        if (!ready || !busy.compareAndSet(false, true)) return false
        synchronized(sessionLock) {
            try {
                if (!ready) return false
                onResult(detectPoses(bitmap), timestampMs)
                return true
            } catch (e: Exception) {
                onError(e.message ?: "YOLO26 pose detect failed")
                return false
            } finally {
                if (!bitmap.isRecycled) bitmap.recycle()
                busy.set(false)
            }
        }
    }

    override fun close() {
        synchronized(sessionLock) {
            closeLocked()
        }
    }

    private fun closeLocked() {
        ready = false
        busy.set(false)
        session?.close()
        session = null
        environment = null
    }

    private fun detectPoses(bitmap: Bitmap): List<PlayerPose> {
        val env = environment ?: return emptyList()
        val ortSession = session ?: return emptyList()
        val letterbox = letterbox(bitmap)
        val input = bitmapToNchw(letterbox.bitmap)
        letterbox.bitmap.recycle()

        val detections = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(input),
            longArrayOf(1L, 3L, inputSize.toLong(), inputSize.toLong())
        ).use { tensor ->
            ortSession.run(mapOf(inputName to tensor)).use { result ->
                val output = result[0].value
                decodeOutput(output, letterbox, bitmap.width, bitmap.height)
            }
        }

        return nonMaxSuppression(detections)
            .take(config.maxPoses)
            .map { it.toPlayerPose(bitmap.width, bitmap.height) }
    }

    private fun letterbox(source: Bitmap): Letterbox {
        val scale = minOf(inputSize.toFloat() / source.width, inputSize.toFloat() / source.height)
        val scaledWidth = (source.width * scale).toInt().coerceAtLeast(1)
        val scaledHeight = (source.height * scale).toInt().coerceAtLeast(1)
        val dx = (inputSize - scaledWidth) / 2f
        val dy = (inputSize - scaledHeight) / 2f

        val scaled = Bitmap.createScaledBitmap(source, scaledWidth, scaledHeight, true)
        val canvasBitmap = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(canvasBitmap)
        canvas.drawColor(Color.rgb(114, 114, 114))
        canvas.drawBitmap(scaled, dx, dy, Paint(Paint.FILTER_BITMAP_FLAG))
        scaled.recycle()
        return Letterbox(canvasBitmap, scale, dx, dy)
    }

    private fun bitmapToNchw(bitmap: Bitmap): FloatArray {
        val pixels = IntArray(inputSize * inputSize)
        bitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        val planeSize = inputSize * inputSize
        val input = FloatArray(3 * planeSize)
        pixels.forEachIndexed { index, pixel ->
            input[index] = (((pixel shr 16) and 0xFF) / 255f)
            input[planeSize + index] = (((pixel shr 8) and 0xFF) / 255f)
            input[planeSize * 2 + index] = ((pixel and 0xFF) / 255f)
        }
        return input
    }

    private fun decodeOutput(
        output: Any,
        letterbox: Letterbox,
        imageWidth: Int,
        imageHeight: Int
    ): List<YoloDetection> {
        val rows = output as Array<Array<FloatArray>>
        return rows[0].mapNotNull { row ->
            if (row.size < 56) return@mapNotNull null
            val hasClassColumn = row.size >= 57
            val confidence = row[4]
            if (confidence < config.confidenceThreshold) return@mapNotNull null
            val keypointStart = if (hasClassColumn) 6 else 5
            val bbox = decodeBbox(row, letterbox, imageWidth, imageHeight) ?: return@mapNotNull null
            val keypoints = (0 until 17).map { index ->
                val offset = keypointStart + index * 3
                val x = unletterboxX(row[offset], letterbox, imageWidth)
                val y = unletterboxY(row[offset + 1], letterbox, imageHeight)
                CocoKeypoint(
                    index = index,
                    x = x,
                    y = y,
                    score = row[offset + 2].coerceIn(0f, 1f)
                )
            }
            if (keypoints.count { it.score >= config.keypointScoreThreshold } < 7) return@mapNotNull null
            YoloDetection(confidence, bbox, keypoints)
        }.sortedByDescending { it.confidence }
    }

    private fun decodeBbox(
        row: FloatArray,
        letterbox: Letterbox,
        imageWidth: Int,
        imageHeight: Int
    ): RectF? {
        val rawLeft = row[0]
        val rawTop = row[1]
        val rawRight = row[2]
        val rawBottom = row[3]
        val looksLikeXyxy = rawRight > rawLeft && rawBottom > rawTop
        val left: Float
        val top: Float
        val right: Float
        val bottom: Float
        if (looksLikeXyxy) {
            left = unletterboxX(rawLeft, letterbox, imageWidth)
            top = unletterboxY(rawTop, letterbox, imageHeight)
            right = unletterboxX(rawRight, letterbox, imageWidth)
            bottom = unletterboxY(rawBottom, letterbox, imageHeight)
        } else {
            val cx = unletterboxX(rawLeft, letterbox, imageWidth)
            val cy = unletterboxY(rawTop, letterbox, imageHeight)
            val width = rawRight / letterbox.scale / imageWidth
            val height = rawBottom / letterbox.scale / imageHeight
            left = cx - width / 2f
            top = cy - height / 2f
            right = cx + width / 2f
            bottom = cy + height / 2f
        }
        val box = RectF(
            left.coerceIn(0f, 1f),
            top.coerceIn(0f, 1f),
            right.coerceIn(0f, 1f),
            bottom.coerceIn(0f, 1f)
        )
        return box.takeIf { it.width() > 0.015f && it.height() > 0.035f }
    }

    private fun unletterboxX(value: Float, letterbox: Letterbox, imageWidth: Int): Float {
        return ((value - letterbox.dx) / letterbox.scale / imageWidth).coerceIn(0f, 1f)
    }

    private fun unletterboxY(value: Float, letterbox: Letterbox, imageHeight: Int): Float {
        return ((value - letterbox.dy) / letterbox.scale / imageHeight).coerceIn(0f, 1f)
    }

    private fun nonMaxSuppression(detections: List<YoloDetection>): List<YoloDetection> {
        val selected = mutableListOf<YoloDetection>()
        detections.forEach { candidate ->
            if (selected.none { iou(it.bbox, candidate.bbox) > config.iouThreshold }) {
                selected += candidate
            }
        }
        return selected
    }

    private fun YoloDetection.toPlayerPose(imageWidth: Int, imageHeight: Int): PlayerPose {
        val byCocoIndex = keypoints.associateBy { it.index }
        val points = MutableList(33) { index ->
            LandmarkPoint(index, 0f, 0f, 0f, 0f, 0f)
        }
        COCO_TO_MEDIAPIPE.forEach { (cocoIndex, mediapipeIndex) ->
            val keypoint = byCocoIndex[cocoIndex] ?: return@forEach
            points[mediapipeIndex] = LandmarkPoint(
                index = mediapipeIndex,
                x = keypoint.x,
                y = keypoint.y,
                z = 0f,
                visibility = keypoint.score,
                presence = keypoint.score
            )
        }
        return PlayerPose(
            bbox = bbox,
            landmarks = points,
            imageWidth = imageWidth,
            imageHeight = imageHeight
        )
    }

    private fun configureInput(ortSession: OrtSession) {
        val firstInput = ortSession.inputInfo.entries.first()
        inputName = firstInput.key
        val tensorInfo = firstInput.value.info as? TensorInfo ?: return
        val shape = tensorInfo.shape
        require(shape.size == 4 && shape[1] == 3L && shape[2] == shape[3]) {
            "Unsupported YOLO26 pose ONNX input shape: ${shape.joinToString("x")}"
        }
        inputSize = shape[2].toInt()
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

    private fun assetExists(assetName: String): Boolean {
        return try {
            context.assets.open(assetName).close()
            true
        } catch (_: IOException) {
            false
        }
    }

    private data class Letterbox(
        val bitmap: Bitmap,
        val scale: Float,
        val dx: Float,
        val dy: Float
    )

    private data class YoloDetection(
        val confidence: Float,
        val bbox: RectF,
        val keypoints: List<CocoKeypoint>
    )

    private data class CocoKeypoint(
        val index: Int,
        val x: Float,
        val y: Float,
        val score: Float
    )

    private companion object {
        val COCO_TO_MEDIAPIPE = mapOf(
            0 to 0,
            5 to 11,
            6 to 12,
            7 to 13,
            8 to 14,
            9 to 15,
            10 to 16,
            11 to 23,
            12 to 24,
            13 to 25,
            14 to 26,
            15 to 27,
            16 to 28
        )
    }
}
