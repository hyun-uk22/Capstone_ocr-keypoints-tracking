package com.example.mugunghwa.camera

import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.example.mugunghwa.game.GameEngine
import com.example.mugunghwa.game.GameState
import com.example.mugunghwa.mediapipe.PoseLandmarkerHelper
import com.example.mugunghwa.ocr.OcrHelper
import com.example.mugunghwa.util.ImageUtils
import com.example.mugunghwa.util.TimeUtils

class CameraAnalyzer(
    private val poseLandmarkerHelper: PoseLandmarkerHelper,
    private val ocrHelper: OcrHelper,
    private val gameEngine: GameEngine
) : ImageAnalysis.Analyzer {
    override fun analyze(imageProxy: ImageProxy) {
        try {
            val bitmap = ImageUtils.rotateBitmapIfNeeded(
                bitmap = ImageUtils.imageProxyToBitmap(imageProxy) ?: return,
                rotationDegrees = imageProxy.imageInfo.rotationDegrees
            )
            val timestampMs = TimeUtils.nowMs()
            val shouldRunOcr = gameEngine.uiState.value.gameState in setOf(
                GameState.READY,
                GameState.GREEN_LIGHT
            ) && gameEngine.tracks.value.isNotEmpty()
            val ocrBitmap = if (shouldRunOcr) {
                bitmap.copy(bitmap.config ?: android.graphics.Bitmap.Config.ARGB_8888, false)
            } else {
                null
            }
            val accepted = poseLandmarkerHelper.detectLiveStream(
                bitmap = bitmap,
                rotationDegrees = 0,
                timestampMs = timestampMs
            )
            if (accepted) {
                ocrBitmap?.let { ocrHelper.maybeRun(it, gameEngine.tracks.value, timestampMs) }
            } else {
                ocrBitmap?.let { if (!it.isRecycled) it.recycle() }
                if (!bitmap.isRecycled) bitmap.recycle()
            }
        } finally {
            imageProxy.close()
        }
    }
}
