package com.example.mugunghwa.tts

import android.content.Context
import android.media.MediaPlayer
import android.os.Handler
import android.os.Looper
import android.speech.tts.TextToSpeech
import java.util.Locale

class EliminationTts(context: Context) {
    private val appContext = context.applicationContext
    private val mainHandler = Handler(Looper.getMainLooper())
    private val pending = mutableListOf<String>()
    private var ready = false
    private var tts: TextToSpeech? = null
    private var player: MediaPlayer? = null

    init {
        tts = TextToSpeech(appContext) { status ->
            mainHandler.post {
                ready = status == TextToSpeech.SUCCESS
                if (ready) {
                    tts?.language = Locale.KOREAN
                    tts?.setSpeechRate(0.70f)
                    tts?.setPitch(0.35f)
                    pending.forEach { speakNow(it) }
                    pending.clear()
                } else {
                    pending.clear()
                }
            }
        }
    }

    fun speakOut(playerNumber: String) {
        val normalizedNumber = normalizeNumber(playerNumber)
        val fallbackText = "${normalizedNumber}번 탈락"
        mainHandler.post {
            if (playRecorded(normalizedNumber)) return@post
            if (!ready) {
                pending += fallbackText
                return@post
            }
            speakNow(fallbackText)
        }
    }

    fun stop() {
        mainHandler.post {
            stopRecorded()
            tts?.stop()
        }
    }

    fun close() {
        mainHandler.post {
            pending.clear()
            stopRecorded()
            tts?.stop()
            tts?.shutdown()
            tts = null
            ready = false
        }
    }

    private fun playRecorded(playerNumber: String): Boolean {
        val number = playerNumber.toIntOrNull() ?: return false
        val resourceId = appContext.resources.getIdentifier("out_$number", "raw", appContext.packageName)
        if (resourceId == 0) return false

        stopRecorded()
        val mediaPlayer = MediaPlayer.create(appContext, resourceId) ?: return false
        player = mediaPlayer
        mediaPlayer.setOnCompletionListener {
            if (player == it) player = null
            it.release()
        }
        mediaPlayer.setOnErrorListener { mp, _, _ ->
            if (player == mp) player = null
            mp.release()
            false
        }
        mediaPlayer.start()
        return true
    }

    private fun stopRecorded() {
        player?.let {
            if (it.isPlaying) it.stop()
            it.release()
        }
        player = null
    }

    private fun speakNow(text: String) {
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "out-${System.currentTimeMillis()}")
    }

    private fun normalizeNumber(playerNumber: String): String {
        return playerNumber.filter { it.isDigit() }.ifBlank { playerNumber }
    }
}
