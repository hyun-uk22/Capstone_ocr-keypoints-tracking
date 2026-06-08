package com.example.mugunghwa.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.mugunghwa.camera.CameraLensMode
import com.example.mugunghwa.game.GameUiState

@Composable
fun GameSettingsPanel(
    uiState: GameUiState,
    greenDuration: LightDurationOption,
    redDuration: LightDurationOption,
    cameraLensMode: CameraLensMode,
    recordGameVideo: Boolean,
    isRecording: Boolean,
    onGreenDurationChange: (LightDurationOption) -> Unit,
    onRedDurationChange: (LightDurationOption) -> Unit,
    onCameraLensModeChange: (CameraLensMode) -> Unit,
    onRecordGameVideoChange: (Boolean) -> Unit,
    onStartAuto: (greenSeconds: Int, redSeconds: Int) -> Unit,
    onStopAuto: () -> Unit,
    onCalibrate: () -> Unit,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .width(214.dp)
            .fillMaxHeight()
            .background(Color(0xE6101014), RoundedCornerShape(topStart = 14.dp, bottomStart = 14.dp))
            .border(1.dp, Color(0x55FFFFFF), RoundedCornerShape(topStart = 14.dp, bottomStart = 14.dp))
            .verticalScroll(rememberScrollState())
            .padding(14.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        SettingsSection(title = "GAME") {
            GameActionControls(
                uiState = uiState,
                greenDuration = greenDuration,
                redDuration = redDuration,
                recordGameVideo = recordGameVideo,
                isRecording = isRecording,
                onRecordGameVideoChange = onRecordGameVideoChange,
                onStartAuto = onStartAuto,
                onStopAuto = onStopAuto,
                onCalibrate = onCalibrate
            )
        }
        SettingsSection(title = "RED") {
            LightDurationControl(
                selected = redDuration,
                onSelected = onRedDurationChange
            )
        }
        SettingsSection(title = "GREEN") {
            LightDurationControl(
                selected = greenDuration,
                onSelected = onGreenDurationChange
            )
        }
        SettingsSection(title = "CAM") {
            CameraSelectorControl(
                selected = cameraLensMode,
                onSelected = onCameraLensModeChange
            )
        }
    }
}

@Composable
private fun GameActionControls(
    uiState: GameUiState,
    greenDuration: LightDurationOption,
    redDuration: LightDurationOption,
    recordGameVideo: Boolean,
    isRecording: Boolean,
    onRecordGameVideoChange: (Boolean) -> Unit,
    onStartAuto: (greenSeconds: Int, redSeconds: Int) -> Unit,
    onStopAuto: () -> Unit,
    onCalibrate: () -> Unit
) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Button(
            onClick = { onStartAuto(greenDuration.resolveSeconds(), redDuration.resolveSeconds()) },
            modifier = Modifier.fillMaxWidth(),
            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF00A884))
        ) { Text(if (uiState.isAutoRunning) "RUNNING" else "START") }
        Button(
            onClick = onStopAuto,
            modifier = Modifier.fillMaxWidth(),
            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFFF2E63))
        ) { Text("STOP") }
        OutlinedButton(
            onClick = { onRecordGameVideoChange(!recordGameVideo) },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text(
                when {
                    isRecording -> "REC..."
                    recordGameVideo -> "REC ON"
                    else -> "REC OFF"
                }
            )
        }
        OutlinedButton(onClick = onCalibrate, modifier = Modifier.fillMaxWidth()) { Text("RECALIB") }
    }
}

@Composable
private fun SettingsSection(
    title: String,
    content: @Composable () -> Unit
) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(
            text = title,
            color = Color(0xFFBDBDBD),
            fontSize = 11.sp,
            fontWeight = FontWeight.Bold
        )
        content()
    }
}
