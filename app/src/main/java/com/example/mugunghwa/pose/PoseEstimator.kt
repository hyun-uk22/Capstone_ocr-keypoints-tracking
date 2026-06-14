package com.example.mugunghwa.pose

import android.graphics.Bitmap

interface PoseEstimator {
    fun setup()
    fun detectLiveStream(bitmap: Bitmap, rotationDegrees: Int, timestampMs: Long): Boolean
    fun close()
}
