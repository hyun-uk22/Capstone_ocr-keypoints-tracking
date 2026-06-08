package com.example.mugunghwa.ui

import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.dp
import com.example.mugunghwa.game.GameState

@Composable
fun YoungheeFace(
    gameState: GameState,
    greenDurationMs: Int,
    modifier: Modifier = Modifier
) {
    val rotation = remember { Animatable(180f) }

    LaunchedEffect(gameState, greenDurationMs) {
        when (gameState) {
            GameState.GREEN_LIGHT -> {
                rotation.snapTo(180f)
                rotation.animateTo(
                    targetValue = 0f,
                    animationSpec = tween(durationMillis = greenDurationMs.coerceAtLeast(800), easing = LinearEasing)
                )
            }
            GameState.RED_LIGHT -> rotation.snapTo(0f)
            GameState.READY, GameState.FINISHED -> rotation.snapTo(180f)
        }
    }

    Box(
        modifier = modifier
            .size(82.dp)
            .clip(CircleShape)
            .background(Color(0xAA101014))
    ) {
        Canvas(Modifier.size(82.dp)) {
            val faceVisible = rotation.value < 90f
            val center = Offset(size.width / 2f, size.height / 2f)
            val skin = Color(0xFFFFD6B0)
            val hair = Color(0xFF17120F)
            val pink = Color(0xFFFF2E63)
            val mint = Color(0xFF00E5A8)

            drawCircle(Color(0xCC000000), radius = 38f, center = center, style = Stroke(3f))
            drawCircle(if (faceVisible) skin else hair, radius = 31f, center = center)

            if (faceVisible) {
                drawArc(hair, startAngle = 190f, sweepAngle = 140f, useCenter = true, topLeft = Offset(13f, 8f), size = androidx.compose.ui.geometry.Size(56f, 40f))
                drawCircle(Color.Black, radius = 3.5f, center = Offset(31f, 40f))
                drawCircle(Color.Black, radius = 3.5f, center = Offset(51f, 40f))
                drawLine(Color.Black, Offset(35f, 55f), Offset(47f, 55f), strokeWidth = 3f)
                drawCircle(pink, radius = 6f, center = Offset(20f, 65f))
                drawCircle(mint, radius = 6f, center = Offset(62f, 65f))
            } else {
                drawCircle(Color(0xFF2A211C), radius = 20f, center = center, style = Stroke(5f))
                drawLine(pink, Offset(24f, 62f), Offset(58f, 62f), strokeWidth = 5f)
            }
        }
    }
}
