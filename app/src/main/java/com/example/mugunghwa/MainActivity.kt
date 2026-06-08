package com.example.mugunghwa

import android.os.Bundle
import androidx.activity.ComponentActivity
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
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.example.mugunghwa.camera.CameraAnalyzer
import com.example.mugunghwa.camera.CameraLensMode
import com.example.mugunghwa.camera.CameraScreen
import com.example.mugunghwa.game.GameEngine
import com.example.mugunghwa.game.GameState
import com.example.mugunghwa.mediapipe.PoseLandmarkerHelper
import com.example.mugunghwa.ocr.OcrHelper
import com.example.mugunghwa.ui.CalibrationCountdownOverlay
import com.example.mugunghwa.ui.GameOverlay
import com.example.mugunghwa.ui.GameSettingsPanel
import com.example.mugunghwa.ui.LightDurationOption
import com.example.mugunghwa.ui.RedLightBorderOverlay
import com.example.mugunghwa.ui.ShapeMenuButton
import com.example.mugunghwa.tts.EliminationTts
import com.example.mugunghwa.tts.SoundtrackPlayer
import com.example.mugunghwa.util.TimeUtils

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        hideSystemBars()

        setContent {
            val gameEngine = remember { GameEngine() }
            val ocrHelper = remember { OcrHelper(gameEngine) }
            val soundtrackPlayer = remember { SoundtrackPlayer(applicationContext) }
            val eliminationTts = remember { EliminationTts(applicationContext) }
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

            DisposableEffect(Unit) {
                gameEngine.setSpeechCallback { soundtrackPlayer.playMugunghwa(flush = true) }
                gameEngine.setEliminationSpeechCallback { playerNumber ->
                    eliminationTts.speakOut(playerNumber)
                }
                poseHelper.setup()
                onDispose {
                    poseHelper.close()
                    ocrHelper.close()
                    soundtrackPlayer.close()
                    eliminationTts.close()
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
                                overlapCount = uiState.overlapCount
                            )
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
                                onGreenDurationChange = { greenDuration = it },
                                onRedDurationChange = { redDuration = it },
                                onCameraLensModeChange = { cameraLensMode = it },
                                onStart = gameEngine::startGame,
                                onStartAuto = gameEngine::startAutoGame,
                                onStopAuto = {
                                    soundtrackPlayer.stop()
                                    eliminationTts.stop()
                                    gameEngine.reset()
                                },
                                onToggle = gameEngine::toggleGreenRed,
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
    overlapCount: Int
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
                isAutoRunning -> "AUTO"
                else -> ""
            },
            color = Color.White,
            style = MaterialTheme.typography.bodyMedium
        )
    }
}
