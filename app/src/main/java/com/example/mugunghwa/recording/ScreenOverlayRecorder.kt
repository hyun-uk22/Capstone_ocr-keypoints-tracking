package com.example.mugunghwa.recording

import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.hardware.display.DisplayManager
import android.media.MediaRecorder
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.net.Uri
import android.os.Build
import android.os.ParcelFileDescriptor
import android.provider.MediaStore
import android.util.Size
import android.view.WindowManager
import java.text.SimpleDateFormat
import java.util.Locale

class ScreenOverlayRecorder(private val context: Context) {
    private val projectionManager =
        context.getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager

    private var mediaProjection: MediaProjection? = null
    private var mediaRecorder: MediaRecorder? = null
    private var outputDescriptor: ParcelFileDescriptor? = null
    private var outputUri: Uri? = null
    private var projectionCallback: MediaProjection.Callback? = null
    private var isStarted = false

    fun createCaptureIntent(): Intent = projectionManager.createScreenCaptureIntent()

    fun start(
        resultCode: Int,
        data: Intent,
        onStateChange: (Boolean) -> Unit,
        onMessage: (String?) -> Unit
    ) {
        if (isStarted) return

        try {
            MediaProjectionForegroundService.start(context)
            val size = captureSize()
            val descriptor = createOutputDescriptor()
            val recorder = createRecorder(size, descriptor.fileDescriptor)
            val projection = projectionManager.getMediaProjection(resultCode, data)
            val callback = object : MediaProjection.Callback() {
                override fun onStop() {
                    stop(onStateChange, onMessage)
                }
            }

            projection.registerCallback(callback, null)
            projection.createVirtualDisplay(
                "FreezeOverlayRecording",
                size.width,
                size.height,
                context.resources.displayMetrics.densityDpi,
                DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
                recorder.surface,
                null,
                null
            )
            recorder.start()

            mediaProjection = projection
            mediaRecorder = recorder
            outputDescriptor = descriptor
            projectionCallback = callback
            isStarted = true
            onStateChange(true)
            onMessage("Screen recording started")
        } catch (e: Exception) {
            cleanup()
            onStateChange(false)
            onMessage("Screen recording failed: ${e.message ?: "unknown error"}")
        }
    }

    fun stop(
        onStateChange: (Boolean) -> Unit = {},
        onMessage: (String?) -> Unit = {}
    ) {
        if (!isStarted && mediaRecorder == null && mediaProjection == null) return

        try {
            mediaRecorder?.stop()
            finalizeOutput()
            onMessage("Saved to Movies/Freeze")
        } catch (_: Exception) {
            outputUri?.let { context.contentResolver.delete(it, null, null) }
            onMessage("Screen recording failed")
        } finally {
            cleanup()
            onStateChange(false)
        }
    }

    private fun createRecorder(size: Size, fileDescriptor: java.io.FileDescriptor): MediaRecorder {
        val recorder = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            MediaRecorder(context)
        } else {
            @Suppress("DEPRECATION")
            MediaRecorder()
        }

        recorder.apply {
            setVideoSource(MediaRecorder.VideoSource.SURFACE)
            setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
            setOutputFile(fileDescriptor)
            setVideoEncoder(MediaRecorder.VideoEncoder.H264)
            setVideoEncodingBitRate(10_000_000)
            setVideoFrameRate(30)
            setVideoSize(size.width, size.height)
            prepare()
        }
        return recorder
    }

    private fun createOutputDescriptor(): ParcelFileDescriptor {
        val nowSeconds = System.currentTimeMillis() / 1000
        val name = "Freeze_overlay_${SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(System.currentTimeMillis())}.mp4"
        val values = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
            put(MediaStore.MediaColumns.MIME_TYPE, "video/mp4")
            put(MediaStore.Video.Media.DATE_ADDED, nowSeconds)
            put(MediaStore.Video.Media.DATE_MODIFIED, nowSeconds)
            put(MediaStore.Video.Media.DATE_TAKEN, System.currentTimeMillis())
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.Video.Media.RELATIVE_PATH, "Movies/Freeze/")
                put(MediaStore.Video.Media.IS_PENDING, 1)
            }
        }
        val uri = context.contentResolver.insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, values)
            ?: error("Cannot create MediaStore video")
        outputUri = uri
        return context.contentResolver.openFileDescriptor(uri, "w")
            ?: error("Cannot open MediaStore video")
    }

    private fun finalizeOutput() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            outputUri?.let { uri ->
                val values = ContentValues().apply {
                    put(MediaStore.Video.Media.IS_PENDING, 0)
                    put(MediaStore.Video.Media.DATE_MODIFIED, System.currentTimeMillis() / 1000)
                }
                context.contentResolver.update(uri, values, null, null)
            }
        }
    }

    private fun captureSize(): Size {
        val windowManager = context.getSystemService(Context.WINDOW_SERVICE) as WindowManager
        val size = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            val bounds = windowManager.currentWindowMetrics.bounds
            Size(bounds.width(), bounds.height())
        } else {
            @Suppress("DEPRECATION")
            android.graphics.Point().also { windowManager.defaultDisplay.getRealSize(it) }
                .let { Size(it.x, it.y) }
        }
        val width = size.width.coerceAtLeast(2).toEven()
        val height = size.height.coerceAtLeast(2).toEven()
        return Size(width, height)
    }

    private fun cleanup() {
        projectionCallback?.let { callback ->
            mediaProjection?.unregisterCallback(callback)
        }
        mediaRecorder?.release()
        mediaProjection?.stop()
        outputDescriptor?.close()
        mediaRecorder = null
        mediaProjection = null
        outputDescriptor = null
        outputUri = null
        projectionCallback = null
        isStarted = false
        MediaProjectionForegroundService.stop(context)
    }

    private fun Int.toEven(): Int = if (this % 2 == 0) this else this - 1
}
