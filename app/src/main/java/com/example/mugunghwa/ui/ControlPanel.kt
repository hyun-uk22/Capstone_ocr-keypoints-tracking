package com.example.mugunghwa.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.getValue
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.mugunghwa.game.GameUiState

@Composable
fun ControlPanel(
    uiState: GameUiState,
    greenDuration: LightDurationOption,
    redDuration: LightDurationOption,
    onStart: () -> Unit,
    onStartAuto: (greenSeconds: Int, redSeconds: Int) -> Unit,
    onStopAuto: () -> Unit,
    onToggle: () -> Unit,
    onCalibrate: () -> Unit,
    onReset: () -> Unit,
    modifier: Modifier = Modifier
) {
    var expanded by remember { mutableStateOf(false) }

    Column(
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 18.dp, vertical = 22.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        if (expanded) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(10.dp))
                    .background(Color(0xDD101014))
                    .border(1.dp, Color(0xFFFF2E63), RoundedCornerShape(10.dp))
                    .padding(12.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = if (uiState.isAutoRunning) "AUTO" else "READY",
                        color = if (uiState.isAutoRunning) Color(0xFF00E5A8) else Color(0xFFFFC400),
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold
                    )
                }

                Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                    Button(
                        onClick = { onStartAuto(greenDuration.resolveSeconds(), redDuration.resolveSeconds()) },
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFFF2E63))
                    ) { Text("Auto") }
                    Button(
                        onClick = onStopAuto,
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF00A884))
                    ) { Text("Stop") }
                    OutlinedButton(onClick = onReset, modifier = Modifier.weight(1f)) { Text("Reset") }
                }

                Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                    OutlinedButton(onClick = onStart, modifier = Modifier.weight(1f)) { Text("Start") }
                    OutlinedButton(onClick = onToggle, modifier = Modifier.weight(1f)) { Text("G/R") }
                    OutlinedButton(onClick = onCalibrate, modifier = Modifier.weight(1f)) { Text("Calib") }
                }
            }
        }

        Row(
            modifier = Modifier
                .padding(top = 10.dp)
                .clip(RoundedCornerShape(24.dp))
                .background(Color(0xCC101014))
                .border(2.dp, Color(0xCCFFFFFF), RoundedCornerShape(24.dp))
                .clickable { expanded = !expanded }
                .padding(horizontal = 18.dp, vertical = 9.dp),
            horizontalArrangement = Arrangement.spacedBy(14.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            ShapeMark("O", Color(0xFFFF2E63))
            ShapeMark("^", Color(0xFF00E5A8))
            ShapeMark("[]", Color.White)
        }
    }
}

@Composable
private fun ShapeMark(text: String, color: Color) {
    Box(
        modifier = Modifier
            .size(36.dp)
            .clip(CircleShape),
        contentAlignment = Alignment.Center
    ) {
        Text(
            text = text,
            color = color,
            fontSize = 25.sp,
            lineHeight = 25.sp,
            fontWeight = FontWeight.Black,
            textAlign = TextAlign.Center
        )
    }
}
