package com.example.mugunghwa.tts

import android.content.Context
import android.media.MediaPlayer
import android.os.Handler
import android.os.Looper
import com.example.mugunghwa.R

class SoundtrackPlayer(context: Context) {
    private val appContext = context.applicationContext
    private val mainHandler = Handler(Looper.getMainLooper())
    private var player: MediaPlayer? = null
    private var stopRunnable: Runnable? = null

    fun playMugunghwa(flush: Boolean = true) {
        mainHandler.post {
            if (flush) stopCurrent()
            val mediaPlayer = MediaPlayer.create(appContext, R.raw.squid_game_soundtrack) ?: return@post
            player = mediaPlayer
            val scheduledStop = Runnable {
                if (player == mediaPlayer) stopCurrent()
            }
            stopRunnable = scheduledStop
            mediaPlayer.setOnCompletionListener {
                if (player == it) player = null
                if (stopRunnable == scheduledStop) stopRunnable = null
                it.release()
            }
            mediaPlayer.setOnErrorListener { mp, _, _ ->
                if (player == mp) player = null
                if (stopRunnable == scheduledStop) stopRunnable = null
                mp.release()
                true
            }
            mediaPlayer.start()
            mainHandler.postDelayed(scheduledStop, 4900L)
        }
    }

    fun close() {
        mainHandler.post { stopCurrent() }
    }

    fun stop() {
        mainHandler.post { stopCurrent() }
    }

    private fun stopCurrent() {
        stopRunnable?.let { mainHandler.removeCallbacks(it) }
        stopRunnable = null
        player?.let {
            if (it.isPlaying) it.stop()
            it.release()
        }
        player = null
    }
}
