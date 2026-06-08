package com.example.mugunghwa.util

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer

object ImageUtils {
    fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        return when (imageProxy.format) {
            ImageFormat.YUV_420_888 -> yuv420ToBitmap(imageProxy)
            else -> rgbaToBitmap(imageProxy)
        }
    }

    fun rotateBitmapIfNeeded(bitmap: Bitmap, rotationDegrees: Int): Bitmap {
        if (rotationDegrees == 0) return bitmap
        val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
        val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        bitmap.recycle()
        return rotated
    }

    private fun rgbaToBitmap(imageProxy: ImageProxy): Bitmap? {
        val plane = imageProxy.planes.firstOrNull() ?: return null
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride.coerceAtLeast(1)
        val rowPixels = rowStride / pixelStride
        val rowBitmap = Bitmap.createBitmap(rowPixels, imageProxy.height, Bitmap.Config.ARGB_8888)
        rowBitmap.copyPixelsFromBuffer(plane.buffer.rewinded())
        return if (rowPixels == imageProxy.width) {
            rowBitmap
        } else {
            val cropped = Bitmap.createBitmap(rowBitmap, 0, 0, imageProxy.width, imageProxy.height)
            rowBitmap.recycle()
            cropped
        }
    }

    private fun yuv420ToBitmap(imageProxy: ImageProxy): Bitmap? {
        val nv21 = yuv420ToNv21(imageProxy)
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, imageProxy.width, imageProxy.height, null)
        val output = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, imageProxy.width, imageProxy.height), 85, output)
        val bytes = output.toByteArray()
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }

    private fun yuv420ToNv21(imageProxy: ImageProxy): ByteArray {
        val yBuffer = imageProxy.planes[0].buffer.rewinded()
        val uBuffer = imageProxy.planes[1].buffer.rewinded()
        val vBuffer = imageProxy.planes[2].buffer.rewinded()
        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()
        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)
        return nv21
    }

    private fun ByteBuffer.rewinded(): ByteBuffer {
        rewind()
        return this
    }
}
