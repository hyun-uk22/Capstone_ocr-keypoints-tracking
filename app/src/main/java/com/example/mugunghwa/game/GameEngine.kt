package com.example.mugunghwa.game

import com.example.mugunghwa.tracking.MovementScorer
import com.example.mugunghwa.tracking.PlayerPose
import com.example.mugunghwa.tracking.PlayerTrack
import com.example.mugunghwa.tracking.TrackManager
import android.graphics.RectF
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlin.math.ceil

data class GameUiState(
    val gameState: GameState = GameState.READY,
    val activePlayerCount: Int = 0,
    val eliminatedCount: Int = 0,
    val totalTrackedCount: Int = 0,
    val threshold: Float = 0.006f,
    val modelMissingMessage: String? = null,
    val isCalibrating: Boolean = false,
    val isAutoRunning: Boolean = false,
    val overlapCount: Int = 0,
    val centerMessage: String? = null,
    val greenDurationMs: Int = 3000
)

private data class EliminationAnnouncement(
    val displayText: String,
    val spokenNumber: String
)

class GameEngine(
    private val trackManager: TrackManager = TrackManager(),
    private val movementScorer: MovementScorer = MovementScorer()
) {
    // TODO: Add an automatic Green/Red timer.
    // TODO: Trigger TTS for "무궁화 꽃이 피었습니다".
    // TODO: Store eliminated players in a local DB.
    // TODO: Replace centroid matching with stronger multi-person re-identification.
    // TODO: Improve OCR accuracy with better name-tag ROI tuning.
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private var autoGameJob: Job? = null
    private var centerMessageJob: Job? = null
    private var eliminationAnnouncementJob: Job? = null
    private var redInterruptedByElimination = false
    private var speechCallback: ((String) -> Unit)? = null
    private var eliminationSpeechCallback: ((String) -> Unit)? = null
    private val eliminationQueue = mutableListOf<EliminationAnnouncement>()
    private val calibrationScores = mutableListOf<Float>()
    private val redBaselineTrackIds = mutableSetOf<Int>()
    private var calibrationEndMs = 0L
    private var redBaselineReadyAtMs = 0L

    private val _uiState = MutableStateFlow(GameUiState())
    val uiState: StateFlow<GameUiState> = _uiState.asStateFlow()

    private val _tracks = MutableStateFlow<List<PlayerTrack>>(emptyList())
    val tracks: StateFlow<List<PlayerTrack>> = _tracks.asStateFlow()

    private val _logs = MutableStateFlow<List<String>>(emptyList())
    val logs: StateFlow<List<String>> = _logs.asStateFlow()

    fun setSpeechCallback(callback: (String) -> Unit) {
        speechCallback = callback
    }

    fun setEliminationSpeechCallback(callback: (String) -> Unit) {
        eliminationSpeechCallback = callback
    }

    fun startGame() {
        stopAutoGame()
        _uiState.value = _uiState.value.copy(gameState = GameState.GREEN_LIGHT)
        log("Game started. GREEN_LIGHT.")
    }

    fun toggleGreenRed() {
        stopAutoGame()
        toggleGreenRedInternal()
    }

    fun startAutoGame(greenSeconds: Int, redSeconds: Int, calibrateFirst: Boolean = true) {
        autoGameJob?.cancel()
        autoGameJob = scope.launch {
            _uiState.value = _uiState.value.copy(isAutoRunning = true)
            log("Auto game started.")
            if (calibrateFirst) {
                val durationMs = 4000L
                startCalibration(System.currentTimeMillis(), durationMs = durationMs)
                for (remaining in 4 downTo 1) {
                    _uiState.value = _uiState.value.copy(centerMessage = "READY $remaining")
                    delay(1000L)
                }
                finishCalibrationNow()
                _uiState.value = _uiState.value.copy(centerMessage = "START")
                delay(450L)
                _uiState.value = _uiState.value.copy(centerMessage = null)
            }
            while (isActive) {
                val greenMs = greenSeconds.coerceAtLeast(5) * 1000
                setGreenLight(greenMs)
                delay(greenMs.toLong())

                redInterruptedByElimination = false
                setRedLight()
                val redEndMs = System.currentTimeMillis() + redSeconds.coerceAtLeast(5) * 1000L
                while (isActive && System.currentTimeMillis() < redEndMs && !redInterruptedByElimination) {
                    delay(100L)
                }
                if (redInterruptedByElimination) {
                    eliminationAnnouncementJob?.join()
                }
            }
        }
    }

    fun stopAutoGame() {
        autoGameJob?.cancel()
        autoGameJob = null
        eliminationAnnouncementJob?.cancel()
        eliminationAnnouncementJob = null
        eliminationQueue.clear()
        redInterruptedByElimination = false
        if (_uiState.value.isAutoRunning) {
            _uiState.value = _uiState.value.copy(isAutoRunning = false)
            log("Auto game stopped.")
        }
    }

    private fun toggleGreenRedInternal() {
        val next = when (_uiState.value.gameState) {
            GameState.READY, GameState.RED_LIGHT -> GameState.GREEN_LIGHT
            GameState.GREEN_LIGHT -> GameState.RED_LIGHT
            GameState.FINISHED -> GameState.FINISHED
        }
        _uiState.value = _uiState.value.copy(gameState = next)
        if (next == GameState.RED_LIGHT) {
            prepareDelayedRedBaseline()
        }
        log("State changed: $next")
    }

    private fun setGreenLight(greenDurationMs: Int = 3000) {
        redBaselineTrackIds.clear()
        redBaselineReadyAtMs = 0L
        _uiState.value = _uiState.value.copy(
            gameState = GameState.GREEN_LIGHT,
            greenDurationMs = greenDurationMs
        )
        speechCallback?.invoke("mugunghwa")
        log("State changed: GREEN_LIGHT")
    }

    private fun setRedLight() {
        prepareDelayedRedBaseline()
        _uiState.value = _uiState.value.copy(gameState = GameState.RED_LIGHT, centerMessage = null)
        log("State changed: RED_LIGHT")
    }

    fun startCalibration(nowMs: Long, durationMs: Long = 7000L) {
        calibrationScores.clear()
        calibrationEndMs = nowMs + durationMs
        _uiState.value = _uiState.value.copy(isCalibrating = true)
        log("Stillness calibration started.")
    }

    private fun finishCalibrationNow() {
        if (!_uiState.value.isCalibrating) return
        finishCalibrationIfNeeded(calibrationEndMs)
    }

    fun onPoses(poses: List<PlayerPose>, timestampMs: Long) {
        val tracks = trackManager.update(poses, timestampMs)
        markOverlaps(tracks)
        val state = _uiState.value.gameState

        tracks.forEach { track ->
            if (track.overlapping) {
                track.ocrCandidates.clear()
            }

            val score = when {
                state == GameState.RED_LIGHT && !ensureRedBaselineReady(track, timestampMs) -> 0f
                state == GameState.RED_LIGHT -> movementScorer.redLightScore(track)
                else -> movementScorer.frameDeltaScore(track)
            }
            if (_uiState.value.isCalibrating) {
                calibrationScores += score
            }

            when (state) {
                GameState.GREEN_LIGHT, GameState.READY -> movementScorer.updateGreenBaseline(track)
                GameState.RED_LIGHT -> {
                    if (track.overlapping) {
                        track.redViolationFrames = 0
                    } else {
                        evaluateRedLight(track, score)
                    }
                }
                GameState.FINISHED -> Unit
            }
        }

        finishCalibrationIfNeeded(timestampMs)
        _tracks.value = tracks.map { it.copy() }
        publishCounts()
    }

    fun lockOcrLabel(trackId: Int, label: String) {
        val track = trackManager.activeTracks().firstOrNull { it.id == trackId } ?: return
        if (!canAcceptOcrLabel(track, label)) return
        track.label = label
        log("OCR label locked: $label")
        _tracks.value = trackManager.activeTracks().map { it.copy() }
    }

    fun recordOcrCandidate(trackId: Int, candidate: String): Boolean {
        val track = trackManager.activeTracks().firstOrNull { it.id == trackId } ?: return false
        if (!canAcceptOcrLabel(track, candidate)) return false
        val count = (track.ocrCandidates[candidate] ?: 0) + 1
        track.ocrCandidates[candidate] = count
        return count >= 3 && canAcceptOcrLabel(track, candidate)
    }

    fun setModelMissingMessage(message: String?) {
        _uiState.value = _uiState.value.copy(modelMissingMessage = message)
    }

    fun reset() {
        stopAutoGame()
        trackManager.reset()
        movementScorer.reset()
        calibrationScores.clear()
        redBaselineTrackIds.clear()
        calibrationEndMs = 0L
        redBaselineReadyAtMs = 0L
        _tracks.value = emptyList()
        _uiState.value = GameUiState()
        _logs.value = emptyList()
        log("Reset completed.")
    }

    private fun evaluateRedLight(track: PlayerTrack, score: Float) {
        if (track.eliminated) return
        if (score > movementScorer.movementThreshold) {
            track.redViolationFrames += 1
            val strongMovement = score > movementScorer.movementThreshold * 3f
            if (strongMovement || track.redViolationFrames >= 3) {
                track.eliminated = true
                showEliminatedMessage(track)
                log("Player ${track.displayName} moved during RED_LIGHT. Eliminated.")
            }
        } else {
            track.redViolationFrames = 0
        }
    }

    private fun showEliminatedMessage(track: PlayerTrack) {
        val playerName = track.label ?: "#${track.id}"
        val spokenNumber = track.label ?: track.id.toString()
        eliminationQueue += EliminationAnnouncement(
            displayText = "$playerName OUT",
            spokenNumber = spokenNumber
        )
        if (_uiState.value.isAutoRunning && _uiState.value.gameState == GameState.RED_LIGHT) {
            redInterruptedByElimination = true
        }
        if (eliminationAnnouncementJob?.isActive == true) return

        centerMessageJob?.cancel()
        eliminationAnnouncementJob = scope.launch {
            while (eliminationQueue.isNotEmpty()) {
                val announcement = eliminationQueue.removeAt(0)
                _uiState.value = _uiState.value.copy(centerMessage = announcement.displayText)
                eliminationSpeechCallback?.invoke(announcement.spokenNumber)
                delay(2500L)
                if (_uiState.value.centerMessage == announcement.displayText) {
                    _uiState.value = _uiState.value.copy(centerMessage = null)
                }
            }

            if (_uiState.value.isAutoRunning && _uiState.value.gameState == GameState.RED_LIGHT) {
                _uiState.value = _uiState.value.copy(centerMessage = "START")
                delay(3000L)
                if (_uiState.value.centerMessage == "START") {
                    _uiState.value = _uiState.value.copy(centerMessage = null)
                }
            }
        }
    }

    private fun finishCalibrationIfNeeded(nowMs: Long) {
        if (!_uiState.value.isCalibrating || nowMs < calibrationEndMs) return
        val p99 = percentile(calibrationScores, 0.99f)
        val threshold = (p99 * 1.6f).coerceIn(0.006f, 0.020f)
        movementScorer.setCalibratedThreshold(threshold)
        _uiState.value = _uiState.value.copy(isCalibrating = false, threshold = threshold)
        log("Calibration threshold updated: ${"%.4f".format(threshold)}")
    }

    private fun publishCounts() {
        val tracks = trackManager.activeTracks()
        _uiState.value = _uiState.value.copy(
            activePlayerCount = tracks.count { !it.eliminated },
            eliminatedCount = tracks.count { it.eliminated },
            totalTrackedCount = trackManager.totalTrackedCount(),
            threshold = movementScorer.movementThreshold,
            overlapCount = tracks.count { it.overlapping }
        )
    }

    private fun log(message: String) {
        _logs.value = (listOf(message) + _logs.value).take(20)
    }

    private fun percentile(values: List<Float>, percentile: Float): Float {
        if (values.isEmpty()) return movementScorer.movementThreshold
        val sorted = values.sorted()
        val index = ceil(percentile * sorted.lastIndex).toInt().coerceIn(0, sorted.lastIndex)
        return sorted[index]
    }

    private fun markOverlaps(tracks: List<PlayerTrack>) {
        tracks.forEach { it.overlapping = false }
        for (i in tracks.indices) {
            for (j in i + 1 until tracks.size) {
                val first = tracks[i]
                val second = tracks[j]
                if (first.missedFrames > 0 || second.missedFrames > 0) continue
                if (first.eliminated || second.eliminated) continue
                if (iou(first.bbox, second.bbox) > 0.38f || containedOverlap(first.bbox, second.bbox) > 0.55f) {
                    first.overlapping = true
                    second.overlapping = true
                }
            }
        }
    }

    private fun prepareDelayedRedBaseline() {
        redBaselineTrackIds.clear()
        redBaselineReadyAtMs = System.currentTimeMillis() + RED_BASELINE_DELAY_MS
        trackManager.activeTracks().forEach { it.redViolationFrames = 0 }
    }

    private fun ensureRedBaselineReady(track: PlayerTrack, timestampMs: Long): Boolean {
        if (track.id in redBaselineTrackIds) return true
        track.redViolationFrames = 0
        if (timestampMs < redBaselineReadyAtMs) {
            movementScorer.updateGreenBaseline(track)
            return false
        }

        movementScorer.captureRedLightBaseline(track)
        redBaselineTrackIds += track.id
        return false
    }

    private fun canAcceptOcrLabel(track: PlayerTrack, label: String): Boolean {
        if (!track.active || track.eliminated || track.overlapping || track.missedFrames > 0) return false
        if (track.label != null) return false
        return !trackManager.hasLabelAssignedToOtherTrack(label, track.id)
    }

    private fun iou(a: RectF, b: RectF): Float {
        val left = maxOf(a.left, b.left)
        val top = maxOf(a.top, b.top)
        val right = minOf(a.right, b.right)
        val bottom = minOf(a.bottom, b.bottom)
        val intersection = (right - left).coerceAtLeast(0f) * (bottom - top).coerceAtLeast(0f)
        val union = a.width() * a.height() + b.width() * b.height() - intersection
        if (union <= 0f) return 0f
        return intersection / union
    }

    private fun containedOverlap(a: RectF, b: RectF): Float {
        val left = maxOf(a.left, b.left)
        val top = maxOf(a.top, b.top)
        val right = minOf(a.right, b.right)
        val bottom = minOf(a.bottom, b.bottom)
        val intersection = (right - left).coerceAtLeast(0f) * (bottom - top).coerceAtLeast(0f)
        val smallerArea = minOf(a.width() * a.height(), b.width() * b.height())
        if (smallerArea <= 0f) return 0f
        return intersection / smallerArea
    }

    companion object {
        private const val RED_BASELINE_DELAY_MS = 700L
    }
}
