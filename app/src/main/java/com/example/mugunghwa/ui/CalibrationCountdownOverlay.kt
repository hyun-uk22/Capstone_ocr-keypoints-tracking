package com.example.mugunghwa.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

@Composable
fun CalibrationCountdownOverlay(
    message: String?,
    isCalibrating: Boolean,
    modifier: Modifier = Modifier
) {
    val display = countdownText(message, isCalibrating) ?: return
    Box(
        modifier = modifier
            .background(Color(0xAA000000), RoundedCornerShape(12.dp))
            .padding(horizontal = 30.dp, vertical = 16.dp)
    ) {
        Text(
            text = display,
            color = Color.White,
            fontSize = 60.sp,
            lineHeight = 60.sp,
            fontWeight = FontWeight.Black
        )
    }
}

private fun countdownText(message: String?, isCalibrating: Boolean): String? {
    if (message == "START") return "START"
    if (!isCalibrating) return message
    val value = message?.substringAfter("READY ", missingDelimiterValue = "")?.toIntOrNull()
    return when (value) {
        4 -> "3"
        3 -> "2"
        2 -> "1"
        else -> null
    }
}
