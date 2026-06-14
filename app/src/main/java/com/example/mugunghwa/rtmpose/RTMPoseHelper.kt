package com.example.mugunghwa.rtmpose

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import android.graphics.RectF
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import com.example.mugunghwa.pose.PoseEstimator
import com.example.mugunghwa.tracking.LandmarkPoint
import com.example.mugunghwa.tracking.PlayerPose
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector
import java.io.IOException
import java.nio.FloatBuffer
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.exp
import kotlin.math.min

data class RTMPoseConfig(
    val modelAssetPath: String = "rtmpose.onnx",
    val objectDetectorAssetPath: String = "efficientdet_lite0.tflite",
    val numPoses: Int = 5,
    val objectScoreThreshold: Float = 0.35f,
    val keypointScoreThreshold: Float = 0.10f
)

class RTMPoseHelper(
    private val context: Context,
    private val config: RTMPoseConfig = RTMPoseConfig(),
    private val onResult: (List<PlayerPose>, Long) -> Unit,
    private val onError: (String) -> Unit
) : PoseEstimator {
    private val sessionLock = Any()
    private val busy = AtomicBoolean(false)
    private var environment: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var objectDetector: ObjectDetector? = null
    private var ready = false
    private var inputName = "input"
    private var inputWidth = 192
    private var inputHeight = 256

    override fun setup() {
        synchronized(sessionLock) {
            if (!assetExists(config.modelAssetPath)) {
                ready = false
                onError("${config.modelAssetPath} is missing. Place the RTMPose ONNX model in app/src/main/assets.")
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

                if (assetExists(config.objectDetectorAssetPath)) {
                    objectDetector = createObjectDetector()
                }
                ready = true
                onError("")
            } catch (e: Exception) {
                closeLocked()
                onError(e.message ?: "RTMPose initialization failed")
            }
        }
    }

    override fun detectLiveStream(bitmap: Bitmap, rotationDegrees: Int, timestampMs: Long): Boolean {
        if (!ready || !busy.compareAndSet(false, true)) return false
        synchronized(sessionLock) {
            try {
                if (!ready) return false
                val poses = detectPoses(bitmap)
                onResult(poses, timestampMs)
                return true
            } catch (e: Exception) {
                onError(e.message ?: "RTMPose detect failed")
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
        objectDetector?.close()
        session = null
        objectDetector = null
        environment = null
    }

    private fun detectPoses(bitmap: Bitmap): List<PlayerPose> {
        val detector = objectDetector
        val cropRects = if (detector != null) {
            detector.detect(BitmapImageBuilder(bitmap).build()).detections()
                .filter { detection ->
                    detection.categories().any {
                        it.categoryName().equals("person", ignoreCase = true) &&
                            it.score() >= config.objectScoreThreshold
                    }
                }
                .sortedByDescending { it.boundingBox().width() * it.boundingBox().height() }
                .take(config.numPoses)
                .map { it.boundingBox().toExpandedCrop(bitmap.width, bitmap.height) }
        } else {
            listOf(Rect(0, 0, bitmap.width, bitmap.height))
        }

        return cropRects.mapNotNull { cropRect ->
            if (cropRect.width() < 24 || cropRect.height() < 24) return@mapNotNull null
            val crop = Bitmap.createBitmap(bitmap, cropRect.left, cropRect.top, cropRect.width(), cropRect.height())
            try {
                runSinglePose(crop, cropRect, bitmap.width, bitmap.height)
            } finally {
                crop.recycle()
            }
        }
    }

    private fun runSinglePose(
        crop: Bitmap,
        cropRect: Rect,
        imageWidth: Int,
        imageHeight: Int
    ): PlayerPose? {
        val env = environment ?: return null
        val ortSession = session ?: return null
        val input = bitmapToNchwInput(crop)
        val shape = longArrayOf(1L, 3L, inputHeight.toLong(), inputWidth.toLong())

        val keypoints = OnnxTensor.createTensor(env, FloatBuffer.wrap(input), shape).use { tensor ->
            ortSession.run(mapOf(inputName to tensor)).use { result ->
                val outputs = result.associate { entry -> entry.key to entry.value.value }
                val outputX = outputs["simcc_x"] ?: outputs.values.elementAtOrNull(0) ?: return null
                val outputY = outputs["simcc_y"] ?: outputs.values.elementAtOrNull(1) ?: return null
                decodeSimcc(outputX, outputY)
            }
        }

        if (keypoints.count { it.score >= config.keypointScoreThreshold } < 7) return null
        return mapCocoKeypoints(keypoints, cropRect, imageWidth, imageHeight)
    }

    private fun bitmapToNchwInput(bitmap: Bitmap): FloatArray {
        val resized = Bitmap.createScaledBitmap(bitmap, inputWidth, inputHeight, true)
        val pixels = IntArray(inputWidth * inputHeight)
        resized.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)
        if (resized !== bitmap) resized.recycle()

        val planeSize = inputWidth * inputHeight
        val input = FloatArray(3 * planeSize)
        pixels.forEachIndexed { index, pixel ->
            val r = ((pixel shr 16) and 0xFF).toFloat()
            val g = ((pixel shr 8) and 0xFF).toFloat()
            val b = (pixel and 0xFF).toFloat()
            input[index] = (r - 123.675f) / 58.395f
            input[planeSize + index] = (g - 116.28f) / 57.12f
            input[planeSize * 2 + index] = (b - 103.53f) / 57.375f
        }
        return input
    }

    private fun decodeSimcc(outputX: Any, outputY: Any): List<CocoKeypoint> {
        val xScores = outputX as Array<Array<FloatArray>>
        val yScores = outputY as Array<Array<FloatArray>>
        val keypointCount = min(17, min(xScores[0].size, yScores[0].size))
        return (0 until keypointCount).map { index ->
            val xBest = argmax(xScores[0][index])
            val yBest = argmax(yScores[0][index])
            CocoKeypoint(
                index = index,
                x = (xBest.index.toFloat() / SIMCC_SPLIT_RATIO / inputWidth).coerceIn(0f, 1f),
                y = (yBest.index.toFloat() / SIMCC_SPLIT_RATIO / inputHeight).coerceIn(0f, 1f),
                score = min(sigmoid(xBest.score), sigmoid(yBest.score))
            )
        }
    }

    private fun mapCocoKeypoints(
        keypoints: List<CocoKeypoint>,
        cropRect: Rect,
        imageWidth: Int,
        imageHeight: Int
    ): PlayerPose {
        val byCocoIndex = keypoints.associateBy { it.index }
        val points = MutableList(33) { index ->
            LandmarkPoint(index, 0f, 0f, 0f, 0f, 0f)
        }
        COCO_TO_MEDIAPIPE.forEach { (cocoIndex, mediapipeIndex) ->
            val keypoint = byCocoIndex[cocoIndex] ?: return@forEach
            val x = (cropRect.left + keypoint.x * cropRect.width()) / imageWidth
            val y = (cropRect.top + keypoint.y * cropRect.height()) / imageHeight
            points[mediapipeIndex] = LandmarkPoint(
                index = mediapipeIndex,
                x = x.coerceIn(0f, 1f),
                y = y.coerceIn(0f, 1f),
                z = 0f,
                visibility = keypoint.score,
                presence = keypoint.score
            )
        }
        return PlayerPose(
            bbox = RectF(
                cropRect.left.toFloat() / imageWidth,
                cropRect.top.toFloat() / imageHeight,
                cropRect.right.toFloat() / imageWidth,
                cropRect.bottom.toFloat() / imageHeight
            ),
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
        require(shape.size == 4 && shape[1] == 3L) {
            "Unsupported RTMPose ONNX input shape: ${shape.joinToString("x")}"
        }
        inputHeight = shape[2].toInt()
        inputWidth = shape[3].toInt()
    }

    private fun createObjectDetector(): ObjectDetector {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath(config.objectDetectorAssetPath)
            .build()
        val options = ObjectDetector.ObjectDetectorOptions.builder()
            .setBaseOptions(baseOptions)
            .setRunningMode(RunningMode.IMAGE)
            .setMaxResults(config.numPoses)
            .setScoreThreshold(config.objectScoreThreshold)
            .setCategoryAllowlist(listOf("person"))
            .build()
        return ObjectDetector.createFromOptions(context, options)
    }

    private fun RectF.toExpandedCrop(imageWidth: Int, imageHeight: Int): Rect {
        val padX = width() * 0.12f
        val padY = height() * 0.10f
        return Rect(
            (left - padX).toInt().coerceIn(0, imageWidth - 1),
            (top - padY).toInt().coerceIn(0, imageHeight - 1),
            (right + padX).toInt().coerceIn(1, imageWidth),
            (bottom + padY).toInt().coerceIn(1, imageHeight)
        )
    }

    private fun argmax(values: FloatArray): BestScore {
        var bestIndex = 0
        var bestScore = Float.NEGATIVE_INFINITY
        values.forEachIndexed { index, score ->
            if (score > bestScore) {
                bestIndex = index
                bestScore = score
            }
        }
        return BestScore(bestIndex, bestScore)
    }

    private fun sigmoid(value: Float): Float {
        return (1.0f / (1.0f + exp(-value))).coerceIn(0f, 1f)
    }

    private fun assetExists(assetName: String): Boolean {
        return try {
            context.assets.open(assetName).close()
            true
        } catch (_: IOException) {
            false
        }
    }

    private data class CocoKeypoint(
        val index: Int,
        val x: Float,
        val y: Float,
        val score: Float
    )

    private data class BestScore(
        val index: Int,
        val score: Float
    )

    private companion object {
        private const val SIMCC_SPLIT_RATIO = 2f

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
