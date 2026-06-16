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
import com.example.mugunghwa.pose.PoseModelType

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun PoseModelSelectorControl(
    selected: PoseModelType,
    onSelected: (PoseModelType) -> Unit,
    modifier: Modifier = Modifier
) {
    FlowRow(
        modifier = modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        PoseModelChip(
            text = "Media",
            selected = selected == PoseModelType.MEDIAPIPE,
            onClick = { onSelected(PoseModelType.MEDIAPIPE) },
        )
        PoseModelChip(
            text = "RTM",
            selected = selected == PoseModelType.RTMPOSE,
            onClick = { onSelected(PoseModelType.RTMPOSE) },
        )
        PoseModelChip(
            text = "YOLO",
            selected = selected == PoseModelType.YOLO26,
            onClick = { onSelected(PoseModelType.YOLO26) },
        )
    }
}

@Composable
private fun PoseModelChip(
    text: String,
    selected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    Text(
        text = text,
        color = Color.White,
        fontSize = 13.sp,
        fontWeight = FontWeight.Bold,
        modifier = modifier
            .background(
                color = if (selected) Color(0xFF00A884) else Color(0x66333338),
                shape = RoundedCornerShape(14.dp)
            )
            .clickable(onClick = onClick)
            .padding(horizontal = 12.dp, vertical = 9.dp)
    )
}
