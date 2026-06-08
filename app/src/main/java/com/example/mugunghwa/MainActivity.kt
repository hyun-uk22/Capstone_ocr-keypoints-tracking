package com.example.mugunghwa

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.example.mugunghwa.camera.CameraAnalyzer
import com.example.mugunghwa.camera.CameraLensMode
import com.example.mugunghwa.camera.CameraScreen
import com.example.mugunghwa.game.GameEngine
import com.example.mugunghwa.game.GameState
import com.example.mugunghwa.mediapipe.PoseLandmarkerHelper
import com.example.mugunghwa.ocr.OcrHelper
import com.example.mugunghwa.recording.MediaProjectionForegroundService
import com.example.mugunghwa.recording.ScreenOverlayRecorder
import com.example.mugunghwa.ui.CalibrationCountdownOverlay
import com.example.mugunghwa.ui.GameOverlay
import com.example.mugunghwa.ui.GameSettingsPanel
import com.example.mugunghwa.ui.LightDurationOption
import com.example.mugunghwa.ui.RedLightBorderOverlay
import com.example.mugunghwa.ui.ShapeMenuButton
import com.example.mugunghwa.tts.EliminationTts
import com.example.mugunghwa.tts.SoundtrackPlayer
import com.example.mugunghwa.util.TimeUtils
import kotlinx.coroutines.delay

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        hideSystemBars()

        setContent {
            val gameEngine = remember { GameEngine() }
            val ocrHelper = remember { OcrHelper(gameEngine) }
            val soundtrackPlayer = remember { SoundtrackPlayer(applicationContext) }
            val eliminationTts = remember { EliminationTts(applicationContext) }
            val screenRecorder = remember { ScreenOverlayRecorder(this) }
            val poseHelper = remember {
                PoseLandmarkerHelper(
                    context = applicationContext,
                    onResult = { poses, timestampMs -> gameEngine.onPoses(poses, timestampMs) },
                    onError = { message ->
                        gameEngine.setModelMissingMessage(message.takeIf { it.isNotBlank() })
                    }
                )
            }
            val analyzer = remember { CameraAnalyzer(poseHelper, ocrHelper, gameEngine) }
            var cameraLensMode by remember { mutableStateOf(CameraLensMode.NORMAL) }
            var settingsOpen by remember { mutableStateOf(false) }
            var greenDuration by remember { mutableStateOf<LightDurationOption>(LightDurationOption.Fixed(5)) }
            var redDuration by remember { mutableStateOf<LightDurationOption>(LightDurationOption.Fixed(5)) }
            var recordGameVideo by remember { mutableStateOf(false) }
            var screenCaptureResultCode by remember { mutableStateOf<Int?>(null) }
            var screenCaptureData by remember { mutableStateOf<Intent?>(null) }
            var isRecording by remember { mutableStateOf(false) }
            var recordingMessage by remember { mutableStateOf<String?>(null) }
            var showOcrModelMessage by remember { mutableStateOf(true) }
            val lifecycleOwner = LocalLifecycleOwner.current
            val screenCaptureLauncher = rememberLauncherForActivityResult(
                ActivityResultContracts.StartActivityForResult()
            ) { result ->
                if (result.resultCode == Activity.RESULT_OK && result.data != null) {
                    screenCaptureResultCode = result.resultCode
                    screenCaptureData = result.data
                    recordGameVideo = true
                    MediaProjectionForegroundService.start(applicationContext)
                    recordingMessage = "Screen recording ready"
                } else {
                    screenCaptureResultCode = null
                    screenCaptureData = null
                    recordGameVideo = false
                    recordingMessage = "Screen recording permission denied"
                }
            }

            LaunchedEffect(Unit) {
                delay(8000L)
                showOcrModelMessage = false
            }

            DisposableEffect(Unit) {
                gameEngine.setSpeechCallback { soundtrackPlayer.playMugunghwa(flush = true) }
                gameEngine.setEliminationSpeechCallback { playerNumber ->
                    eliminationTts.speakOut(playerNumber)
                }
                poseHelper.setup()
                onDispose {
                    screenRecorder.stop()
                    MediaProjectionForegroundService.stop(applicationContext)
                    poseHelper.close()
                    ocrHelper.close()
                    soundtrackPlayer.close()
                    eliminationTts.close()
                }
            }

            DisposableEffect(lifecycleOwner) {
                val observer = LifecycleEventObserver { _, event ->
                    if (event == Lifecycle.Event.ON_STOP) {
                        screenRecorder.stop(
                            onStateChange = { isRecording = it },
                            onMessage = { recordingMessage = it }
                        )
                        MediaProjectionForegroundService.stop(applicationContext)
                        recordingMessage = null
                        soundtrackPlayer.stop()
                        eliminationTts.stop()
                        gameEngine.reset()
                    }
                }
                lifecycleOwner.lifecycle.addObserver(observer)
                onDispose {
                    lifecycleOwner.lifecycle.removeObserver(observer)
                }
            }

            val uiState = gameEngine.uiState.collectAsStateWithLifecycle().value
            val tracks = gameEngine.tracks.collectAsStateWithLifecycle().value

            MaterialTheme {
                Surface(Modifier.fillMaxSize(), color = Color.Black) {
                    Box(Modifier.fillMaxSize()) {
                        CameraScreen(
                            analyzer = analyzer,
                            lensMode = cameraLensMode
                        )
                        GameOverlay(tracks = tracks)
                        RedLightBorderOverlay(
                            isRedLight = uiState.gameState == GameState.RED_LIGHT,
                            isGreenLight = uiState.gameState == GameState.GREEN_LIGHT
                        )
                        CalibrationCountdownOverlay(
                            message = uiState.centerMessage,
                            isCalibrating = uiState.isCalibrating,
                            modifier = Modifier.align(Alignment.Center)
                        )

                        Column(
                            modifier = Modifier
                                .align(Alignment.TopCenter)
                                .fillMaxWidth()
                        ) {
                            StatusBar(
                                active = uiState.activePlayerCount,
                                isCalibrating = uiState.isCalibrating,
                                isAutoRunning = uiState.isAutoRunning,
                                overlapCount = uiState.overlapCount,
                                isRecording = isRecording
                            )
                            recordingMessage?.let { message ->
                                Text(
                                    text = message,
                                    color = Color.White,
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .background(Color(0xAA101014))
                                        .padding(10.dp)
                                )
                            }
                            if (showOcrModelMessage) {
                                Text(
                                    text = "OCR model may download on first run",
                                    color = Color.White,
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .background(Color(0xAA101014))
                                        .padding(10.dp)
                                )
                            }
                            uiState.modelMissingMessage?.let { message ->
                                if (message.isNotBlank()) {
                                    Text(
                                        text = "MODEL REQUIRED",
                                        color = Color.White,
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .background(Color(0xCCB00020))
                                            .padding(10.dp)
                                    )
                                }
                            }
                        }

                        if (settingsOpen) {
                            GameSettingsPanel(
                                uiState = uiState,
                                greenDuration = greenDuration,
                                redDuration = redDuration,
                                cameraLensMode = cameraLensMode,
                                recordGameVideo = recordGameVideo,
                                isRecording = isRecording,
                                onGreenDurationChange = { greenDuration = it },
                                onRedDurationChange = { redDuration = it },
                                onCameraLensModeChange = { cameraLensMode = it },
                                onRecordGameVideoChange = { enabled ->
                                    if (enabled) {
                                        screenCaptureLauncher.launch(screenRecorder.createCaptureIntent())
                                    } else {
                                        recordGameVideo = false
                                        screenCaptureResultCode = null
                                        screenCaptureData = null
                                        if (isRecording) {
                                            screenRecorder.stop(
                                                onStateChange = { isRecording = it },
                                                onMessage = { recordingMessage = it }
                                            )
                                        }
                                        MediaProjectionForegroundService.stop(applicationContext)
                                    }
                                },
                                onStartAuto = { greenSeconds, redSeconds ->
                                    settingsOpen = false
                                    recordingMessage = null
                                    if (recordGameVideo) {
                                        val resultCode = screenCaptureResultCode
                                        val data = screenCaptureData
                                        if (resultCode != null && data != null) {
                                            screenRecorder.start(
                                                resultCode = resultCode,
                                                data = data,
                                                onStateChange = { isRecording = it },
                                                onMessage = { recordingMessage = it }
                                            )
                                        } else {
                                            recordGameVideo = false
                                            recordingMessage = "Turn REC ON again"
                                        }
                                    }
                                    gameEngine.startAutoGame(greenSeconds, redSeconds)
                                },
                                onStopAuto = {
                                    screenRecorder.stop(
                                        onStateChange = { isRecording = it },
                                        onMessage = { recordingMessage = it }
                                    )
                                    screenCaptureResultCode = null
                                    screenCaptureData = null
                                    recordGameVideo = false
                                    MediaProjectionForegroundService.stop(applicationContext)
                                    soundtrackPlayer.stop()
                                    eliminationTts.stop()
                                    gameEngine.reset()
                                },
                                onCalibrate = { gameEngine.startCalibration(TimeUtils.nowMs()) },
                                modifier = Modifier.align(Alignment.CenterEnd)
                            )
                        }
                        ShapeMenuButton(
                            onClick = { settingsOpen = !settingsOpen },
                            modifier = Modifier
                                .align(Alignment.CenterEnd)
                                .padding(end = if (settingsOpen) 224.dp else 10.dp)
                        )
                    }
                }
            }
        }
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        if (hasFocus) hideSystemBars()
    }

    private fun hideSystemBars() {
        WindowCompat.setDecorFitsSystemWindows(window, false)
        WindowInsetsControllerCompat(window, window.decorView).apply {
            systemBarsBehavior = WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
            hide(WindowInsetsCompat.Type.systemBars())
        }
    }
}

@androidx.compose.runtime.Composable
private fun StatusBar(
    active: Int,
    isCalibrating: Boolean,
    isAutoRunning: Boolean,
    overlapCount: Int,
    isRecording: Boolean
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(Color(0xAA000000))
            .padding(horizontal = 10.dp, vertical = 8.dp),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text("PLAYERS $active", color = Color.White, style = MaterialTheme.typography.bodyMedium)
        Text(
            text = when {
                isCalibrating -> "..."
                overlapCount > 0 -> "! $overlapCount"
                isRecording -> "REC"
                isAutoRunning -> "RUN"
                else -> ""
            },
            color = Color.White,
            style = MaterialTheme.typography.bodyMedium
        )
    }
}
