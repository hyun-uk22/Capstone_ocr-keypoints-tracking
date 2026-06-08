package com.example.mugunghwa.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.dp

@Composable
fun ShapeMenuButton(
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    Box(
        modifier = modifier
            .size(width = 42.dp, height = 110.dp)
            .background(Color(0x99101014), RoundedCornerShape(21.dp))
            .border(1.5.dp, Color(0xAAFFFFFF), RoundedCornerShape(21.dp))
            .clickable(onClick = onClick),
        contentAlignment = Alignment.Center
    ) {
        Canvas(Modifier.size(width = 90.dp, height = 300.dp)) {
            val stroke = Stroke(width = 8.1f)
            val centerX = size.width / 2f
            val centerY = size.height / 2f
            val side = 37.8f
            val halfSide = side / 2f
            val triangleHeight = 32.7f
            val radius = 18.9f
            val shapeGap = 26.1f
            val triangleTop = centerY - triangleHeight * 2f / 3f
            val triangleBase = centerY + triangleHeight / 3f
            drawCircle(
                color = Color(0xFFFF2E63),
                radius = radius,
                center = Offset(centerX, triangleTop - shapeGap - radius),
                style = stroke
            )
            val triangle = Path().apply {
                moveTo(centerX, triangleTop)
                lineTo(centerX - halfSide, triangleBase)
                lineTo(centerX + halfSide, triangleBase)
                close()
            }
            drawPath(triangle, color = Color(0xFF00E5A8), style = stroke)
            drawRect(
                color = Color.White,
                topLeft = Offset(centerX - halfSide, triangleBase + shapeGap),
                size = androidx.compose.ui.geometry.Size(side, side),
                style = stroke
            )
        }
    }
}
