package com.example.freeze.dev

import android.app.Activity
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.view.Gravity
import android.view.View
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.roundToInt

class MainActivity : ComponentActivity() {
    private val mainHandler = Handler(Looper.getMainLooper())
    private val worker = Executors.newSingleThreadExecutor()
    private val stopRequested = AtomicBoolean(false)

    private lateinit var statusText: TextView
    private lateinit var resultText: TextView
    private lateinit var pickButton: Button
    private lateinit var runButton: Button
    private lateinit var stopButton: Button

    private var selectedVideoUri: Uri? = null

    private val pickVideo = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri == null) return@registerForActivityResult
        contentResolver.takePersistableUriPermission(uri, android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION)
        selectedVideoUri = uri
        statusText.text = "Selected video:\n$uri"
        resultText.text = ""
        runButton.isEnabled = true
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(createContentView())
    }

    override fun onDestroy() {
        stopRequested.set(true)
        worker.shutdownNow()
        super.onDestroy()
    }

    private fun createContentView(): View {
        val density = resources.displayMetrics.density
        fun dp(value: Int): Int = (value * density).roundToInt()

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(18), dp(18), dp(18), dp(18))
            gravity = Gravity.CENTER_HORIZONTAL
        }

        val title = TextView(this).apply {
            text = "Freeze Dev\nPose Video Benchmark"
            textSize = 22f
            gravity = Gravity.CENTER
            setPadding(0, 0, 0, dp(14))
        }
        root.addView(title)

        pickButton = Button(this).apply {
            text = "Select Gallery Video"
            setOnClickListener { pickVideo.launch(arrayOf("video/*")) }
        }
        root.addView(pickButton, LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)

        runButton = Button(this).apply {
            text = "Run Benchmark"
            isEnabled = false
            setOnClickListener { selectedVideoUri?.let(::startBenchmark) }
        }
        root.addView(runButton, LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)

        stopButton = Button(this).apply {
            text = "Stop"
            isEnabled = false
            setOnClickListener { stopRequested.set(true) }
        }
        root.addView(stopButton, LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)

        statusText = TextView(this).apply {
            text = "Select a video from gallery."
            textSize = 14f
            setPadding(0, dp(16), 0, dp(10))
        }
        root.addView(statusText)

        resultText = TextView(this).apply {
            textSize = 14f
            setTextIsSelectable(true)
        }
        root.addView(resultText)

        return ScrollView(this).apply { addView(root) }
    }

    private fun startBenchmark(uri: Uri) {
        stopRequested.set(false)
        pickButton.isEnabled = false
        runButton.isEnabled = false
        stopButton.isEnabled = true
        resultText.text = ""
        statusText.text = "Running..."

        worker.execute {
            val result = runCatching { benchmarkVideo(uri) }
                .getOrElse { error -> BenchmarkResult(errorMessage = error.message ?: error.toString()) }
            mainHandler.post {
                pickButton.isEnabled = true
                runButton.isEnabled = selectedVideoUri != null
                stopButton.isEnabled = false
                statusText.text = if (result.errorMessage == null) "Done." else "Failed."
                resultText.text = result.format()
            }
        }
    }

    private fun benchmarkVideo(uri: Uri): BenchmarkResult {
        val retriever = MediaMetadataRetriever()
        retriever.setDataSource(this, uri)

        val durationMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
            ?.toLongOrNull()
            ?: error("Cannot read video duration")
        val sourceWidth = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)
            ?.toIntOrNull()
            ?: 0
        val sourceHeight = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)
            ?.toIntOrNull()
            ?: 0
        val sourceFps = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_CAPTURE_FRAMERATE)
            ?.toDoubleOrNull()
            ?.takeIf { it > 0.0 }
            ?: DEFAULT_FPS
        val estimatedFrameCount = ((durationMs / 1000.0) * sourceFps).roundToInt().coerceAtLeast(1)

        var decodedFrames = 0
        var inferenceAttempts = 0
        var poseSuccessFrames = 0
        var totalInferenceMs = 0L
        var maxInferenceMs = 0L
        var delegateName = "GPU"

        createLandmarker(Delegate.GPU).getOrElse {
            delegateName = "CPU fallback"
            createLandmarker(Delegate.CPU).getOrThrow()
        }.use { landmarker ->
            for (frameIndex in 0 until estimatedFrameCount) {
                if (stopRequested.get()) break

                val timeUs = (frameIndex * 1_000_000.0 / sourceFps).toLong()
                val frame = retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST) ?: continue
                decodedFrames += 1

                val inputBitmap = resizeForInference(frame, MAX_INFERENCE_WIDTH)
                val mpImage = BitmapImageBuilder(inputBitmap).build()
                val timestampMs = (timeUs / 1000).coerceAtLeast(0L)

                val startedAtMs = SystemClock.elapsedRealtime()
                val poseResult = landmarker.detectForVideo(mpImage, timestampMs)
                val inferenceMs = SystemClock.elapsedRealtime() - startedAtMs

                inferenceAttempts += 1
                totalInferenceMs += inferenceMs
                maxInferenceMs = maxOf(maxInferenceMs, inferenceMs)
                if (poseResult.landmarks().isNotEmpty()) {
                    poseSuccessFrames += 1
                }

                if (inputBitmap !== frame) inputBitmap.recycle()
                frame.recycle()

                if (inferenceAttempts % PROGRESS_INTERVAL == 0) {
                    val progress = inferenceAttempts * 100.0 / estimatedFrameCount
                    val avgMs = totalInferenceMs.toDouble() / inferenceAttempts
                    postProgress(inferenceAttempts, estimatedFrameCount, progress, avgMs)
                }
            }
        }
        retriever.release()

        return BenchmarkResult(
            delegateName = delegateName,
            sourceWidth = sourceWidth,
            sourceHeight = sourceHeight,
            sourceFps = sourceFps,
            durationMs = durationMs,
            estimatedFrames = estimatedFrameCount,
            decodedFrames = decodedFrames,
            inferenceAttempts = inferenceAttempts,
            poseSuccessFrames = poseSuccessFrames,
            avgInferenceMs = if (inferenceAttempts == 0) 0.0 else totalInferenceMs.toDouble() / inferenceAttempts,
            maxInferenceMs = maxInferenceMs,
            stopped = stopRequested.get()
        )
    }

    private fun createLandmarker(delegate: Delegate): Result<PoseLandmarker> {
        return runCatching {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath(MODEL_ASSET_PATH)
                .setDelegate(delegate)
                .build()
            val options = PoseLandmarker.PoseLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.VIDEO)
                .setNumPoses(NUM_POSES)
                .setMinPoseDetectionConfidence(0.32f)
                .setMinPosePresenceConfidence(0.30f)
                .setMinTrackingConfidence(0.30f)
                .build()
            PoseLandmarker.createFromOptions(this, options)
        }
    }

    private fun resizeForInference(bitmap: Bitmap, maxWidth: Int): Bitmap {
        if (bitmap.width <= maxWidth) return bitmap
        val scale = maxWidth.toFloat() / bitmap.width
        val height = (bitmap.height * scale).roundToInt().coerceAtLeast(1)
        return Bitmap.createScaledBitmap(bitmap, maxWidth, height, true)
    }

    private fun postProgress(attempts: Int, total: Int, progress: Double, avgMs: Double) {
        mainHandler.post {
            statusText.text = String.format(
                Locale.US,
                "Running... %d / %d (%.1f%%)\nAverage pose update: %.2f ms",
                attempts,
                total,
                progress,
                avgMs
            )
        }
    }

    private data class BenchmarkResult(
        val delegateName: String = "",
        val sourceWidth: Int = 0,
        val sourceHeight: Int = 0,
        val sourceFps: Double = 0.0,
        val durationMs: Long = 0L,
        val estimatedFrames: Int = 0,
        val decodedFrames: Int = 0,
        val inferenceAttempts: Int = 0,
        val poseSuccessFrames: Int = 0,
        val avgInferenceMs: Double = 0.0,
        val maxInferenceMs: Long = 0L,
        val stopped: Boolean = false,
        val errorMessage: String? = null
    ) {
        fun format(): String {
            if (errorMessage != null) return "Error:\n$errorMessage"
            val successRate = if (inferenceAttempts == 0) {
                0.0
            } else {
                poseSuccessFrames.toDouble() / inferenceAttempts
            }
            val updateHz = if (avgInferenceMs <= 0.0) 0.0 else 1000.0 / avgInferenceMs
            return String.format(
                Locale.US,
                "delegate: %s\n" +
                    "video: %dx%d, %.2f fps, %.2f sec\n" +
                    "estimated_frames: %d\n" +
                    "decoded_frames: %d\n" +
                    "inference_attempts: %d\n" +
                    "pose_success_frames: %d\n" +
                    "pose_success_rate: %.3f\n" +
                    "avg_pose_update_ms: %.2f\n" +
                    "max_pose_update_ms: %d\n" +
                    "theoretical_pose_updates_per_sec: %.2f\n" +
                    "stopped: %s",
                delegateName,
                sourceWidth,
                sourceHeight,
                sourceFps,
                durationMs / 1000.0,
                estimatedFrames,
                decodedFrames,
                inferenceAttempts,
                poseSuccessFrames,
                successRate,
                avgInferenceMs,
                maxInferenceMs,
                updateHz,
                stopped
            )
        }
    }

    companion object {
        private const val MODEL_ASSET_PATH = "pose_landmarker_heavy.task"
        private const val NUM_POSES = 5
        private const val DEFAULT_FPS = 30.0
        private const val MAX_INFERENCE_WIDTH = 1280
        private const val PROGRESS_INTERVAL = 30
    }
}
