package com.example.mugunghwa.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color

@Composable
fun RedLightBorderOverlay(
    isRedLight: Boolean,
    isGreenLight: Boolean = false,
    modifier: Modifier = Modifier
) {
    if (!isRedLight && !isGreenLight) return

    Canvas(modifier = modifier.fillMaxSize()) {
        val alertColor = if (isRedLight) Color(0xFFFF1744) else Color(0xFF00E5A8)
        val transparent = Color.Transparent
        val edge = 38f

        drawRect(
            brush = Brush.verticalGradient(listOf(alertColor.copy(alpha = 0.42f), transparent), startY = 0f, endY = edge),
            topLeft = Offset.Zero,
            size = androidx.compose.ui.geometry.Size(size.width, edge)
        )
        drawRect(
            brush = Brush.verticalGradient(listOf(transparent, alertColor.copy(alpha = 0.34f)), startY = size.height - edge, endY = size.height),
            topLeft = Offset(0f, size.height - edge),
            size = androidx.compose.ui.geometry.Size(size.width, edge)
        )
        drawRect(
            brush = Brush.horizontalGradient(listOf(alertColor.copy(alpha = 0.34f), transparent), startX = 0f, endX = edge),
            topLeft = Offset.Zero,
            size = androidx.compose.ui.geometry.Size(edge, size.height)
        )
        drawRect(
            brush = Brush.horizontalGradient(listOf(transparent, alertColor.copy(alpha = 0.34f)), startX = size.width - edge, endX = size.width),
            topLeft = Offset(size.width - edge, 0f),
            size = androidx.compose.ui.geometry.Size(edge, size.height)
        )
    }
}
