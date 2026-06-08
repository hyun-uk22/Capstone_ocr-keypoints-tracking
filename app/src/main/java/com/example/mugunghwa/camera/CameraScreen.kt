package com.example.mugunghwa.camera

import android.Manifest
import android.content.ContentValues
import android.content.Context
import android.content.pm.PackageManager
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.os.Build
import android.provider.MediaStore
import android.util.Size
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.core.CameraInfo
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.UseCaseGroup
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.FallbackStrategy
import androidx.camera.video.MediaStoreOutputOptions
import androidx.camera.video.Quality
import androidx.camera.video.QualitySelector
import androidx.camera.video.Recorder
import androidx.camera.video.Recording
import androidx.camera.video.VideoCapture
import androidx.camera.video.VideoRecordEvent
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import java.text.SimpleDateFormat
import java.util.Locale
import java.util.concurrent.Executors

@Composable
fun CameraScreen(
    analyzer: ImageAnalysis.Analyzer,
    lensMode: CameraLensMode,
    recordingRequested: Boolean,
    onRecordingStateChange: (Boolean) -> Unit,
    onRecordingMessage: (String?) -> Unit,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    var hasCameraPermission by remember {
        mutableStateOf(ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
    }
    var hasStoragePermission by remember {
        mutableStateOf(hasLegacyStoragePermission(context))
    }
    var videoCapture by remember { mutableStateOf<VideoCapture<Recorder>?>(null) }
    var activeRecording by remember { mutableStateOf<Recording?>(null) }
    val permissionLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        hasCameraPermission = granted
    }
    val storagePermissionLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        hasStoragePermission = granted
    }

    LaunchedEffect(Unit) {
        if (!hasCameraPermission) permissionLauncher.launch(Manifest.permission.CAMERA)
    }

    LaunchedEffect(recordingRequested, videoCapture, hasCameraPermission, hasStoragePermission) {
        val capture = videoCapture
        if (!recordingRequested) {
            activeRecording?.stop()
            activeRecording = null
            onRecordingStateChange(false)
            return@LaunchedEffect
        }
        if (!hasCameraPermission || capture == null || activeRecording != null) return@LaunchedEffect
        if (!hasStoragePermission && Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
            storagePermissionLauncher.launch(Manifest.permission.WRITE_EXTERNAL_STORAGE)
            return@LaunchedEffect
        }
        activeRecording = startVideoRecording(
            context = context,
            videoCapture = capture,
            onRecordingStateChange = onRecordingStateChange,
            onRecordingMessage = onRecordingMessage,
            onFinalized = { activeRecording = null }
        )
    }

    DisposableEffect(lensMode) {
        onDispose {
            activeRecording?.stop()
            activeRecording = null
            onRecordingStateChange(false)
        }
    }

    if (!hasCameraPermission) {
        Box(modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Button(onClick = { permissionLauncher.launch(Manifest.permission.CAMERA) }) {
                Text("Grant Camera Permission")
            }
        }
        return
    }

    val analysisExecutor = remember { Executors.newSingleThreadExecutor() }
    DisposableEffect(Unit) {
        onDispose { analysisExecutor.shutdown() }
    }

    key(lensMode) {
        AndroidView(
            modifier = modifier.fillMaxSize(),
            factory = { ctx ->
                val previewView = PreviewView(ctx).apply {
                    scaleType = PreviewView.ScaleType.FILL_CENTER
                }
                val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)
                cameraProviderFuture.addListener({
                    val cameraProvider = cameraProviderFuture.get()
                    val preview = Preview.Builder().build().also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }
                    val recorder = Recorder.Builder()
                        .setQualitySelector(
                            QualitySelector.from(
                                Quality.SD,
                                FallbackStrategy.lowerQualityOrHigherThan(Quality.SD)
                            )
                        )
                        .build()
                    val boundVideoCapture = VideoCapture.withOutput(recorder)
                    val imageAnalysis = ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .setTargetResolution(Size(1280, 720))
                        .build()
                        .also { it.setAnalyzer(analysisExecutor, analyzer) }

                    cameraProvider.unbindAll()
                    val useCaseGroupBuilder = UseCaseGroup.Builder()
                        .addUseCase(preview)
                        .addUseCase(imageAnalysis)
                        .addUseCase(boundVideoCapture)
                    previewView.viewPort?.let { useCaseGroupBuilder.setViewPort(it) }
                    cameraProvider.bindToLifecycle(
                        lifecycleOwner,
                        selectBackCamera(ctx, lensMode),
                        useCaseGroupBuilder.build()
                    )
                    videoCapture = boundVideoCapture
                }, ContextCompat.getMainExecutor(ctx))
                previewView
            }
        )
    }
}

private fun hasLegacyStoragePermission(context: Context): Boolean {
    return Build.VERSION.SDK_INT > Build.VERSION_CODES.P ||
        ContextCompat.checkSelfPermission(context, Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED
}

private fun startVideoRecording(
    context: Context,
    videoCapture: VideoCapture<Recorder>,
    onRecordingStateChange: (Boolean) -> Unit,
    onRecordingMessage: (String?) -> Unit,
    onFinalized: () -> Unit
): Recording {
    val name = "Freeze_${SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(System.currentTimeMillis())}"
    val contentValues = ContentValues().apply {
        put(MediaStore.MediaColumns.DISPLAY_NAME, name)
        put(MediaStore.MediaColumns.MIME_TYPE, "video/mp4")
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            put(MediaStore.Video.Media.RELATIVE_PATH, "Movies/Freeze")
        }
    }
    val outputOptions = MediaStoreOutputOptions.Builder(
        context.contentResolver,
        MediaStore.Video.Media.EXTERNAL_CONTENT_URI
    ).setContentValues(contentValues).build()

    return videoCapture.output
        .prepareRecording(context, outputOptions)
        .start(ContextCompat.getMainExecutor(context)) { event ->
            when (event) {
                is VideoRecordEvent.Start -> {
                    onRecordingStateChange(true)
                    onRecordingMessage("Recording started")
                }
                is VideoRecordEvent.Finalize -> {
                    onRecordingStateChange(false)
                    onFinalized()
                    if (event.hasError()) {
                        onRecordingMessage("Recording failed")
                    } else {
                        onRecordingMessage("Saved to Movies/Freeze")
                    }
                }
            }
        }
}

private fun selectBackCamera(context: Context, lensMode: CameraLensMode): CameraSelector {
    val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    return CameraSelector.Builder()
        .requireLensFacing(CameraSelector.LENS_FACING_BACK)
        .addCameraFilter { cameraInfos ->
            val ranked = cameraInfos.sortedWith(compareBy<CameraInfo> { cameraInfo ->
                focalLength(cameraManager, cameraInfo) ?: Float.MAX_VALUE
            })
            val selected = when (lensMode) {
                CameraLensMode.NORMAL -> normalBackCamera(ranked)
                CameraLensMode.TELE -> ranked.lastOrNull()
            }
            selected?.let { listOf(it) } ?: cameraInfos
        }
        .build()
}

private fun focalLength(cameraManager: CameraManager, cameraInfo: CameraInfo): Float? {
    return try {
        val cameraId = Camera2CameraInfo.from(cameraInfo).cameraId
        cameraManager.getCameraCharacteristics(cameraId)
            .get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
            ?.minOrNull()
    } catch (_: Exception) {
        null
    }
}

private fun normalBackCamera(ranked: List<CameraInfo>): CameraInfo? {
    if (ranked.isEmpty()) return null
    return when (ranked.size) {
        1 -> ranked.first()
        2 -> ranked.first()
        else -> ranked[ranked.size / 2]
    }
}
