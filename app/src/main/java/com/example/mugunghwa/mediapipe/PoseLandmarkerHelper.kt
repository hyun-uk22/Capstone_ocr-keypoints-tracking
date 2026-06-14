package com.example.mugunghwa.mediapipe

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import android.graphics.RectF
import com.example.mugunghwa.pose.PoseEstimator
import com.example.mugunghwa.tracking.LandmarkPoint
import com.example.mugunghwa.tracking.PlayerPose
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import java.io.IOException
import java.util.concurrent.atomic.AtomicBoolean

data class PoseConfig(
    val preferredModelAssetPath: String = "pose_landmarker_heavy.task",
    val fallbackModelAssetPath: String = "pose_landmarker_full.task",
    val lastFallbackModelAssetPath: String = "pose_landmarker_lite.task",
    val objectDetectorAssetPath: String = "efficientdet_lite0.tflite",
    val usePersonCropPipeline: Boolean = true,
    val numPoses: Int = 5,
    val minPoseDetectionConfidence: Float = 0.32f,
    val minPosePresenceConfidence: Float = 0.30f,
    val minTrackingConfidence: Float = 0.30f,
    val objectScoreThreshold: Float = 0.35f
)

class PoseLandmarkerHelper(
    private val context: Context,
    private val config: PoseConfig = PoseConfig(),
    private val onResult: (List<PlayerPose>, Long) -> Unit,
    private val onError: (String) -> Unit
) : PoseEstimator {
    private var fullFrameLandmarker: PoseLandmarker? = null
    private var cropLandmarker: PoseLandmarker? = null
    private var objectDetector: ObjectDetector? = null
    private val busy = AtomicBoolean(false)
    private var inFlightBitmap: Bitmap? = null
    private var inFlightWidth: Int = 0
    private var inFlightHeight: Int = 0

    override fun setup() {
        val poseModelAssetPaths = resolvePoseModelAssetPaths()
        if (poseModelAssetPaths.isEmpty()) {
            onError(
                "${config.preferredModelAssetPath}, ${config.fallbackModelAssetPath}, or ${config.lastFallbackModelAssetPath} is missing. " +
                    "Place a pose landmarker .task model in app/src/main/assets."
            )
            return
        }

        val errors = mutableListOf<String>()
        poseModelAssetPaths.forEach { poseModelAssetPath ->
            closeLandmarkers()
            try {
                if (config.usePersonCropPipeline && assetExists(config.objectDetectorAssetPath)) {
                    try {
                        objectDetector = createObjectDetector()
                        cropLandmarker = createCropLandmarker(poseModelAssetPath)
                    } catch (cropError: Exception) {
                        errors += "$poseModelAssetPath crop: ${cropError.message ?: "crop pipeline failed"}"
                        objectDetector?.close()
                        cropLandmarker?.close()
                        objectDetector = null
                        cropLandmarker = null
                    }
                }
                fullFrameLandmarker = createFullFrameLandmarker(poseModelAssetPath, Delegate.GPU)
                onError("")
                return
            } catch (gpuError: Exception) {
                fullFrameLandmarker?.close()
                fullFrameLandmarker = null
                try {
                    fullFrameLandmarker = createFullFrameLandmarker(poseModelAssetPath, Delegate.CPU)
                    onError("")
                    return
                } catch (cpuError: Exception) {
                    errors += "$poseModelAssetPath: ${cpuError.message ?: gpuError.message ?: "initialization failed"}"
                }
            }
        }

        closeLandmarkers()
        onError(errors.lastOrNull() ?: "PoseLandmarker initialization failed")
    }

    override fun detectLiveStream(bitmap: Bitmap, rotationDegrees: Int, timestampMs: Long): Boolean {
        if (!busy.compareAndSet(false, true)) return false

        val detector = objectDetector
        val cropPose = cropLandmarker
        if (detector != null && cropPose != null) {
            detectPeopleThenSinglePose(bitmap, detector, cropPose, timestampMs)
            return true
        }

        val poseLandmarker = fullFrameLandmarker
        if (poseLandmarker == null) {
            busy.set(false)
            return false
        }

        return try {
            inFlightBitmap = bitmap
            inFlightWidth = bitmap.width
            inFlightHeight = bitmap.height
            val image = BitmapImageBuilder(bitmap).build()
            val imageOptions = ImageProcessingOptions.builder()
                .setRotationDegrees(rotationDegrees)
                .build()
            poseLandmarker.detectAsync(image, imageOptions, timestampMs)
            true
        } catch (e: Exception) {
            busy.set(false)
            recycleInFlightBitmap()
            onError(e.message ?: "PoseLandmarker detect failed")
            false
        }
    }

    override fun close() {
        closeLandmarkers()
        busy.set(false)
        recycleInFlightBitmap()
    }

    private fun closeLandmarkers() {
        fullFrameLandmarker?.close()
        cropLandmarker?.close()
        objectDetector?.close()
        fullFrameLandmarker = null
        cropLandmarker = null
        objectDetector = null
    }

    private fun detectPeopleThenSinglePose(
        bitmap: Bitmap,
        detector: ObjectDetector,
        poseLandmarker: PoseLandmarker,
        timestampMs: Long
    ) {
        try {
            val image = BitmapImageBuilder(bitmap).build()
            val detections = detector.detect(image).detections()
                .filter { detection ->
                    detection.categories().any {
                        it.categoryName().equals("person", ignoreCase = true) &&
                            it.score() >= config.objectScoreThreshold
                    }
                }
                .sortedByDescending { it.boundingBox().width() * it.boundingBox().height() }
                .take(config.numPoses)

            val poses = detections.mapNotNull { detection ->
                val cropRect = detection.boundingBox().toExpandedCrop(bitmap.width, bitmap.height)
                if (cropRect.width() < 24 || cropRect.height() < 24) return@mapNotNull null

                val crop = Bitmap.createBitmap(bitmap, cropRect.left, cropRect.top, cropRect.width(), cropRect.height())
                try {
                    val cropResult = poseLandmarker.detect(BitmapImageBuilder(crop).build())
                    mapCropPose(cropResult, cropRect, bitmap.width, bitmap.height)
                } finally {
                    crop.recycle()
                }
            }

            onResult(poses, timestampMs)
        } catch (e: Exception) {
            onError(e.message ?: "Person crop pose pipeline failed")
        } finally {
            if (!bitmap.isRecycled) bitmap.recycle()
            busy.set(false)
        }
    }

    private fun mapCropPose(
        result: PoseLandmarkerResult,
        cropRect: Rect,
        imageWidth: Int,
        imageHeight: Int
    ): PlayerPose? {
        val landmarks = result.landmarks().firstOrNull() ?: return null
        val cropNorm = RectF(
            cropRect.left.toFloat() / imageWidth,
            cropRect.top.toFloat() / imageHeight,
            cropRect.right.toFloat() / imageWidth,
            cropRect.bottom.toFloat() / imageHeight
        )

        val points = landmarks.mapIndexed { index, landmark ->
            LandmarkPoint(
                index = index,
                x = cropNorm.left + landmark.x() * cropNorm.width(),
                y = cropNorm.top + landmark.y() * cropNorm.height(),
                z = landmark.z(),
                visibility = landmark.visibility().orElse(1f),
                presence = landmark.presence().orElse(1f)
            )
        }

        return PlayerPose(
            bbox = cropNorm,
            landmarks = points,
            imageWidth = imageWidth,
            imageHeight = imageHeight
        )
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

    private fun createCropLandmarker(modelAssetPath: String): PoseLandmarker {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath(modelAssetPath)
            .build()
        val options = PoseLandmarker.PoseLandmarkerOptions.builder()
            .setBaseOptions(baseOptions)
            .setRunningMode(RunningMode.IMAGE)
            .setNumPoses(1)
            .setMinPoseDetectionConfidence(config.minPoseDetectionConfidence)
            .setMinPosePresenceConfidence(config.minPosePresenceConfidence)
            .setMinTrackingConfidence(config.minTrackingConfidence)
            .build()
        return PoseLandmarker.createFromOptions(context, options)
    }

    private fun createFullFrameLandmarker(
        modelAssetPath: String,
        delegate: Delegate
    ): PoseLandmarker {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath(modelAssetPath)
            .setDelegate(delegate)
            .build()

        val options = PoseLandmarker.PoseLandmarkerOptions.builder()
            .setBaseOptions(baseOptions)
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setNumPoses(config.numPoses)
            .setMinPoseDetectionConfidence(config.minPoseDetectionConfidence)
            .setMinPosePresenceConfidence(config.minPosePresenceConfidence)
            .setMinTrackingConfidence(config.minTrackingConfidence)
            .setResultListener { result: PoseLandmarkerResult, _ ->
                val width = inFlightWidth
                val height = inFlightHeight
                busy.set(false)
                recycleInFlightBitmap()
                onResult(PoseResultMapper.map(result, width, height), result.timestampMs())
            }
            .setErrorListener { error ->
                busy.set(false)
                recycleInFlightBitmap()
                onError(error.message ?: "PoseLandmarker error")
            }
            .build()

        return PoseLandmarker.createFromOptions(context, options)
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

    private fun recycleInFlightBitmap() {
        inFlightBitmap?.let {
            if (!it.isRecycled) it.recycle()
        }
        inFlightBitmap = null
        inFlightWidth = 0
        inFlightHeight = 0
    }

    private fun resolvePoseModelAssetPaths(): List<String> {
        return listOf(
            config.preferredModelAssetPath,
            config.fallbackModelAssetPath,
            config.lastFallbackModelAssetPath
        ).distinct().filter { assetExists(it) }
    }

    private fun assetExists(assetName: String): Boolean {
        return try {
            context.assets.open(assetName).close()
            true
        } catch (_: IOException) {
            false
        }
    }

}
