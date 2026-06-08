package com.example.mugunghwa.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

sealed class LightDurationOption {
    data class Fixed(val seconds: Int) : LightDurationOption()
    data object RandomFiveToTen : LightDurationOption()
}

fun LightDurationOption.resolveSeconds(): Int {
    return when (this) {
        is LightDurationOption.Fixed -> seconds
        LightDurationOption.RandomFiveToTen -> (5..10).random()
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun LightDurationControl(
    selected: LightDurationOption,
    onSelected: (LightDurationOption) -> Unit,
    modifier: Modifier = Modifier
) {
    FlowRow(
        modifier = modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(6.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        (5..10).forEach { seconds ->
            DurationChip(
                text = "${seconds}s",
                selected = selected == LightDurationOption.Fixed(seconds),
                onClick = { onSelected(LightDurationOption.Fixed(seconds)) }
            )
        }
        DurationChip(
            text = "R 5-10",
            selected = selected == LightDurationOption.RandomFiveToTen,
            onClick = { onSelected(LightDurationOption.RandomFiveToTen) }
        )
    }
}

@Composable
private fun DurationChip(
    text: String,
    selected: Boolean,
    onClick: () -> Unit
) {
    Text(
        text = text,
        color = Color.White,
        fontSize = 12.sp,
        fontWeight = FontWeight.Bold,
        modifier = Modifier
            .background(
                color = if (selected) Color(0xFFFF2E63) else Color(0x66333338),
                shape = RoundedCornerShape(14.dp)
            )
            .clickable(onClick = onClick)
            .padding(horizontal = 10.dp, vertical = 7.dp)
    )
}
