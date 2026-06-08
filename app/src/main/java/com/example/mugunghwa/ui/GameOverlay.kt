package com.example.mugunghwa.ui

import android.graphics.Paint
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb
import com.example.mugunghwa.tracking.PlayerTrack
import kotlin.math.max

@Composable
fun GameOverlay(
    tracks: List<PlayerTrack>,
    modifier: Modifier = Modifier
) {
    Canvas(modifier = modifier.fillMaxSize()) {
        val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.White.toArgb()
            textSize = 34f
            strokeWidth = 4f
        }
        val outPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.Red.toArgb()
            textSize = 48f
            isFakeBoldText = true
        }
        tracks.forEach { track ->
            val color = when {
                track.eliminated -> Color(0xFFFF1744)
                track.overlapping -> Color(0xFFFFC400)
                else -> Color(0xFF00E5A8)
            }
            val mapper = PreviewCoordinateMapper(
                viewWidth = size.width,
                viewHeight = size.height,
                imageWidth = track.imageWidth,
                imageHeight = track.imageHeight
            )
            val topLeft = mapper.map(track.bbox.left, track.bbox.top)
            val bottomRight = mapper.map(track.bbox.right, track.bbox.bottom)
            val left = topLeft.x
            val top = topLeft.y
            val right = bottomRight.x
            val bottom = bottomRight.y

            drawRect(
                color = color,
                topLeft = Offset(left, top),
                size = Size(right - left, bottom - top),
                style = Stroke(width = 4f)
            )
            drawSkeleton(track, color, mapper)

            val label = "${track.displayName}  ${"%.3f".format(track.latestMovementScore)}"
            drawContext.canvas.nativeCanvas.drawText(label, left, (top - 12f).coerceAtLeast(36f), textPaint)
            if (track.eliminated) {
                drawContext.canvas.nativeCanvas.drawText("OUT", left + 8f, top + 54f, outPaint)
            }
        }
    }
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawSkeleton(
    track: PlayerTrack,
    color: Color,
    mapper: PreviewCoordinateMapper
) {
    val points = track.landmarks.associateBy { it.index }
    SKELETON_CONNECTIONS.forEach { (a, b) ->
        val start = points[a]
        val end = points[b]
        if (start != null && end != null && start.isDrawable(track) && end.isDrawable(track)) {
            drawLine(
                color = color,
                start = mapper.map(start.x, start.y),
                end = mapper.map(end.x, end.y),
                strokeWidth = 4f
            )
        }
    }
    points.values.forEach { point ->
        if (point.index in DRAWABLE_LANDMARKS && point.isDrawable(track)) {
            drawCircle(color = color, radius = 5f, center = mapper.map(point.x, point.y))
        }
    }
}

private fun com.example.mugunghwa.tracking.LandmarkPoint.isDrawable(track: PlayerTrack): Boolean {
    if (visibility < 0.25f || presence < 0.25f) return false
    if (x !in 0f..1f || y !in 0f..1f) return false

    val marginX = (track.bbox.width() * 0.35f).coerceAtLeast(0.04f)
    val marginY = (track.bbox.height() * 0.35f).coerceAtLeast(0.04f)
    val allowedLeft = (track.bbox.left - marginX).coerceAtLeast(0f)
    val allowedTop = (track.bbox.top - marginY).coerceAtLeast(0f)
    val allowedRight = (track.bbox.right + marginX).coerceAtMost(1f)
    val allowedBottom = (track.bbox.bottom + marginY).coerceAtMost(1f)
    return x in allowedLeft..allowedRight && y in allowedTop..allowedBottom
}

private class PreviewCoordinateMapper(
    private val viewWidth: Float,
    private val viewHeight: Float,
    imageWidth: Int,
    imageHeight: Int
) {
    private val safeImageWidth = imageWidth.takeIf { it > 0 } ?: 1
    private val safeImageHeight = imageHeight.takeIf { it > 0 } ?: 1
    private val scale = max(viewWidth / safeImageWidth, viewHeight / safeImageHeight)
    private val dx = (viewWidth - safeImageWidth * scale) / 2f
    private val dy = (viewHeight - safeImageHeight * scale) / 2f

    fun map(normalizedX: Float, normalizedY: Float): Offset {
        return Offset(
            x = normalizedX * safeImageWidth * scale + dx,
            y = normalizedY * safeImageHeight * scale + dy
        )
    }
}

private val SKELETON_CONNECTIONS = listOf(
    11 to 12,
    11 to 13,
    13 to 15,
    12 to 14,
    14 to 16,
    11 to 23,
    12 to 24,
    23 to 24,
    23 to 25,
    25 to 27,
    24 to 26,
    26 to 28,
    0 to 11,
    0 to 12
)

private val DRAWABLE_LANDMARKS = setOf(
    0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28
)
