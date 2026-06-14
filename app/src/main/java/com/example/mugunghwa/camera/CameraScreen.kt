package com.example.mugunghwa.camera

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
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
import java.util.concurrent.Executors

@Composable
fun CameraScreen(
    analyzer: ImageAnalysis.Analyzer,
    lensMode: CameraLensMode,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    var hasCameraPermission by remember {
        mutableStateOf(ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
    }
    val permissionLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        hasCameraPermission = granted
    }

    LaunchedEffect(Unit) {
        if (!hasCameraPermission) permissionLauncher.launch(Manifest.permission.CAMERA)
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

    key(lensMode, analyzer) {
        AndroidView(
            modifier = modifier.fillMaxSize(),
            factory = { ctx ->
                val previewView = PreviewView(ctx).apply {
                    scaleType = PreviewView.ScaleType.FILL_CENTER
                    implementationMode = PreviewView.ImplementationMode.COMPATIBLE
                }
                val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)
                cameraProviderFuture.addListener({
                    val cameraProvider = cameraProviderFuture.get()
                    val preview = Preview.Builder()
                        .setTargetResolution(Size(1920, 1080))
                        .build()
                        .also {
                            it.setSurfaceProvider(previewView.surfaceProvider)
                        }
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
                    previewView.viewPort?.let { useCaseGroupBuilder.setViewPort(it) }
                    cameraProvider.bindToLifecycle(
                        lifecycleOwner,
                        selectBackCamera(ctx, lensMode),
                        useCaseGroupBuilder.build()
                    )
                }, ContextCompat.getMainExecutor(ctx))
                previewView
            }
        )
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
