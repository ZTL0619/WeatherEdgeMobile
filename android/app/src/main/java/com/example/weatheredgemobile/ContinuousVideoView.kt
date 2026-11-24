package com.example.weatheredgemobile

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.media.MediaPlayer
import android.net.Uri
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.os.SystemClock
import android.view.Gravity
import android.view.PixelCopy
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.View
import org.opencv.android.OpenCVLoader
import android.widget.FrameLayout
import java.util.concurrent.ExecutorService
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * v9: Bounded latest-frame scheduling + sparse optical-flow delay compensation.
 *
 * Unlike v4, there is no TextureView.getBitmap() or work on every display frame.
 * Playback works even when detection is disabled. This is an emulator-friendly
 * integration design, NOT a zero-copy or NPU production pipeline.
 */
class ContinuousVideoView(
    context: Context,
    private val worker: ExecutorService,
    private val analyzeFrame: (Bitmap) -> Detection,
    private val onDetection: (Detection, Long, Double) -> Unit,
    private val onPlaybackDone: () -> Unit,
    private val onPlaybackHealth: (Long, Double) -> Unit,
    private val onFailure: (String) -> Unit,
    private val className: (Int) -> String
) : FrameLayout(context), SurfaceHolder.Callback {

    data class Detection(
        val boxes: List<Yolo26Detector.Box>,
        val sourceWidth: Int,
        val sourceHeight: Int,
        val preprocessMs: Double,
        val inferenceMs: Double,
        val pipelineMs: Double,
        val copyMs: Double = 0.0
    )

    private val main = Handler(Looper.getMainLooper())
    private val copyThread = HandlerThread("WeatherEdge-PixelCopy").apply { start() }
    private val copyHandler = Handler(copyThread.looper)
    private val trackingThread = HandlerThread("WeatherEdge-OpticalFlow").apply { start() }
    private val trackingHandler = Handler(trackingThread.looper)
    private val tracker = SparseFlowTracker()
    private val openCvReady = try { OpenCVLoader.initLocal() } catch (_: Throwable) { false }
    private val copyInFlight = AtomicBoolean(false)
    private val inferenceInFlight = AtomicBoolean(false)
    private val surface = SurfaceView(context)
    private val overlay = DetectionOverlay(context, className)
    private var player: MediaPlayer? = null
    private var selectedUri: Uri? = null
    private var surfaceReady = false
    @Volatile private var prepared = false
    @Volatile private var paused = true
    @Volatile private var finished = false
    private var videoWidth = 0
    private var videoHeight = 0
    private var lastDetectionAt = 0L
    private var lastOverlayRefreshAt = 0L
    private var lastCopyErrorAt = 0L
    private var playbackMeasureStart = 0L
    private var playbackPositionStart = 0L
    var playbackRate = 0.0
        private set
    @Volatile private var generation = 0L
    // Incremented whenever playback is paused/resumed. Results from an earlier
    // state must never be painted onto the current video frame.
    @Volatile private var viewEpoch = 0L
    @Volatile private var detectionEnabled = true
    var lastResultAgeMs = 0L
        private set
    var lastDetectionWasStale = false
        private set

    // Target ~16.7 tracking samples/s; one outstanding PixelCopy/flow task and
    // one outstanding detector task. There is no inference-frame FIFO.
    private val captureIntervalMs = 100L
    private var lastCaptureRequestedAt = 0L // main thread only

    private fun isCurrentFrame(token: Long, epoch: Long, pausedSnapshot: Boolean): Boolean =
        token == generation && epoch == viewEpoch && prepared && detectionEnabled &&
            paused == pausedSnapshot && (pausedSnapshot || !finished)

    // Wake the tick as soon as inference is free, subject to the capture rate limit.
    // This removes the extra idle wait without allowing a backlog of callbacks.
    private fun requestFreshFrame() {
        main.post {
            if (!prepared || paused || finished || !detectionEnabled ||
                !surfaceReady || !openCvReady || visibility != View.VISIBLE) return@post
            if (copyInFlight.get()) return@post // the outstanding frame will be consumed
            val delay = (captureIntervalMs -
                (SystemClock.elapsedRealtime() - lastCaptureRequestedAt)).coerceAtLeast(0L)
            main.removeCallbacks(captureLoop)
            main.postDelayed(captureLoop, delay)
        }
    }
    private val maxCaptureEdge = 640

    val isPaused: Boolean get() = paused
    val isFinished: Boolean get() = finished
    val isDetectionEnabled: Boolean get() = detectionEnabled

    // Old boxes are not held stationary. They are forwarded by optical flow on
    // each captured frame; tracks with insufficient visual evidence disappear.

    private fun playbackPosition(): Long = try {
        player?.currentPosition?.toLong() ?: 0L
    } catch (_: IllegalStateException) { 0L }

    private fun clearOverlay() {
        lastOverlayRefreshAt = 0L
        overlay.clear()
    }

    // When the user presses Pause, capture exactly the now-frozen video frame.
    // Wait until any earlier in-flight inference finishes so the paused-frame
    // snapshot is not silently dropped.
    private fun inspectPausedFrame(token: Long, epoch: Long) {
        if (token != generation || epoch != viewEpoch || !paused || !prepared ||
            !detectionEnabled || !surfaceReady || visibility != View.VISIBLE) return
        if (!copyInFlight.get() && !inferenceInFlight.get() &&
            copyInFlight.compareAndSet(false, true)) {
            copyCurrentFrame(pausedSnapshot = true, capturedEpoch = epoch)
        } else {
            main.postDelayed({ inspectPausedFrame(token, epoch) }, 90L)
        }
    }

    private val captureLoop = object : Runnable {
        override fun run() {
            if (!prepared || paused || finished || visibility != View.VISIBLE) return
            updatePlaybackRate()
            // PixelCopy can stall while the video keeps playing. Never leave an
            // old box on screen when no fresh optical-flow update arrives.
            if (detectionEnabled && lastOverlayRefreshAt != 0L &&
                SystemClock.elapsedRealtime() - lastOverlayRefreshAt > 400L) {
                clearOverlay()
            }
            val sinceCapture = SystemClock.elapsedRealtime() - lastCaptureRequestedAt
            if (detectionEnabled && surfaceReady && openCvReady &&
                sinceCapture >= captureIntervalMs && copyInFlight.compareAndSet(false, true)) {
                copyCurrentFrame(pausedSnapshot = false, capturedEpoch = viewEpoch)
            }
            main.postDelayed(this, if (detectionEnabled) captureIntervalMs else 250L)
        }
    }

    init {
        setBackgroundColor(Color.BLACK)
        addView(surface, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT, Gravity.CENTER))
        addView(overlay, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT, Gravity.CENTER))
        surface.holder.addCallback(this)
        if (!openCvReady) main.post { onFailure("OpenCV unavailable: video tracking disabled") }
        addOnLayoutChangeListener { _, _, _, _, _, _, _, _, _ -> updateAspectLayout() }
    }

    /** Call from UI thread. */
    fun play(uri: Uri) {
        stopPlayback()
        selectedUri = uri
        paused = false
        finished = false
        visibility = View.VISIBLE
        if (surfaceReady) preparePlayer()
    }

    fun setDetectionEnabled(enabled: Boolean) {
        if (detectionEnabled == enabled) return
        viewEpoch++ // invalidate work from before the toggle, even after a quick OFF/ON
        detectionEnabled = enabled
        lastDetectionAt = 0L
        lastResultAgeMs = 0L
        lastDetectionWasStale = false
        clearOverlay()
        trackingHandler.post { tracker.reset() }
        if (enabled && prepared && !paused) startCaptureLoop()
        if (enabled && prepared && paused) {
            inspectPausedFrame(generation, viewEpoch)
        }
    }

    // inspectFrame = true only when the user explicitly presses Pause Video.
    // Other callers (e.g. opening a file picker) do not need a costly snapshot.
    fun pausePlayback(inspectFrame: Boolean = false) {
        paused = true
        viewEpoch++
        lastDetectionAt = 0L
        main.removeCallbacks(captureLoop)
        clearOverlay()
        trackingHandler.post { tracker.reset() }
        try { if (prepared && player?.isPlaying == true) player?.pause() }
        catch (_: IllegalStateException) { }
        if (inspectFrame && prepared && detectionEnabled) {
            val token = generation
            val epoch = viewEpoch
            main.postDelayed({ inspectPausedFrame(token, epoch) }, 100L)
        }
    }

    fun resumePlayback() {
        if (finished) { replay(); return }
        viewEpoch++
        paused = false
        lastDetectionAt = 0L
        lastCaptureRequestedAt = 0L
        clearOverlay()
        trackingHandler.post { tracker.reset() }
        try {
            if (prepared) {
                player?.start()
                resetPlaybackRate()
                startCaptureLoop()
                }
        } catch (e: IllegalStateException) { onFailure("Video resume error: ${e.message}") }
    }

    fun replay() {
        selectedUri?.let { play(it) }
    }

    fun stopPlayback() {
        generation++
        viewEpoch++
        selectedUri = null
        main.removeCallbacks(captureLoop)
        prepared = false
        paused = true
        finished = false
        videoWidth = 0
        videoHeight = 0
        lastDetectionAt = 0L
        lastCaptureRequestedAt = 0L
        lastResultAgeMs = 0L
        lastDetectionWasStale = false
        playbackRate = 0.0
        trackingHandler.post { tracker.reset() }
        try { player?.release() } catch (_: Exception) { }
        player = null
        clearOverlay()
        // Older in-flight predictions are ignored via generation/viewEpoch guards.
    }

    /** Release the dedicated copy thread when the Activity is destroyed. */
    fun dispose() {
        stopPlayback()
        surface.holder.removeCallback(this)
        trackingHandler.post { tracker.reset() }
        copyThread.quitSafely()
        trackingThread.quitSafely()
    }

    private fun preparePlayer() {
        val uri = selectedUri ?: return
        if (!surfaceReady || player != null) return
        val token = generation
        val media = MediaPlayer()
        try {
            media.setOnVideoSizeChangedListener { _, w, h ->
                if (token != generation) return@setOnVideoSizeChangedListener
                videoWidth = w
                videoHeight = h
                updateAspectLayout()
            }
            media.setOnPreparedListener { mp ->
                if (token != generation) return@setOnPreparedListener
                prepared = true
                videoWidth = mp.videoWidth
                videoHeight = mp.videoHeight
                updateAspectLayout()
                if (!paused) {
                    mp.start()
                    resetPlaybackRate()
                    startCaptureLoop()
                }
            }
            media.setOnCompletionListener {
                if (token == generation) {
                    finished = true
                    paused = true
                    viewEpoch++
                    main.removeCallbacks(captureLoop)
                    trackingHandler.post { tracker.reset() }
                    clearOverlay()
                    onPlaybackDone()
                }
            }
            media.setOnErrorListener { _, what, extra ->
                if (token == generation) {
                    paused = true
                    viewEpoch++
                    main.removeCallbacks(captureLoop)
                    trackingHandler.post { tracker.reset() }
                    clearOverlay()
                    onFailure("MediaPlayer error $what / $extra")
                }
                true
            }
            media.setVolume(0f, 0f)
            media.setDataSource(context, uri)
            media.setDisplay(surface.holder)
            player = media
            media.prepareAsync()
        } catch (e: Exception) {
            try { media.release() } catch (_: Exception) { }
            if (player === media) player = null
            onFailure("Cannot open video: ${e.message}")
        }
    }

    private fun resetPlaybackRate() {
        playbackMeasureStart = SystemClock.elapsedRealtime()
        playbackPositionStart = try { player?.currentPosition?.toLong() ?: 0L }
                                catch (_: IllegalStateException) { 0L }
        playbackRate = 0.0
    }

    private fun updatePlaybackRate() {
        val wallNow = SystemClock.elapsedRealtime()
        val elapsed = wallNow - playbackMeasureStart
        if (elapsed < 1000L) return
        val videoPosition = try { player?.currentPosition?.toLong() ?: 0L }
                            catch (_: IllegalStateException) { 0L }
        playbackRate = (videoPosition - playbackPositionStart).coerceAtLeast(0L).toDouble() / elapsed
        playbackMeasureStart = wallNow
        playbackPositionStart = videoPosition
        onPlaybackHealth(videoPosition, playbackRate)
    }

    private fun startCaptureLoop() {
        main.removeCallbacks(captureLoop)
        if (prepared && !paused && !finished && visibility == View.VISIBLE) {
            main.post(captureLoop)
        }
    }

    /** Asynchronous PixelCopy; optic-flow thread updates boxes, inference worker runs YOLO. */
    private fun copyCurrentFrame(pausedSnapshot: Boolean = false, capturedEpoch: Long = viewEpoch) {
        val mp = player
        if (mp == null || !prepared || !surface.holder.surface.isValid) {
            copyInFlight.set(false); return
        }
        val w = videoWidth
        val h = videoHeight
        if (w <= 0 || h <= 0 || surface.width <= 0 || surface.height <= 0) {
            copyInFlight.set(false); return
        }
        val ratio = min(1f, maxCaptureEdge.toFloat() / max(w, h))
        val copyW = (w * ratio).roundToInt().coerceAtLeast(1)
        val copyH = (h * ratio).roundToInt().coerceAtLeast(1)
        val token = generation
        val started = SystemClock.elapsedRealtime()
        lastCaptureRequestedAt = started
        val position = playbackPosition()  // Approximate PTS; PixelCopy doesn't expose decoder PTS.
        val bitmap = Bitmap.createBitmap(copyW, copyH, Bitmap.Config.ARGB_8888)
        try {
            PixelCopy.request(surface, bitmap, { status ->
                if (status != PixelCopy.SUCCESS || !isCurrentFrame(token, capturedEpoch, pausedSnapshot)) {
                    bitmap.recycle()
                    copyInFlight.set(false)
                    if (status != PixelCopy.SUCCESS && token == generation) reportCopyError(status)
                    return@request
                }
                val copyMs = (SystemClock.elapsedRealtime() - started).toDouble()
                val frameWidth = bitmap.width
                val frameHeight = bitmap.height
                // All tracker mutations run on one thread. The frame is either
                // recycled here, or handed to the single inference worker (no Bitmap.copy).
                val posted = trackingHandler.post frameTask@{
                    var handedToWorker = false
                    try {
                        if (!isCurrentFrame(token, capturedEpoch, pausedSnapshot)) return@frameTask
                        if (!pausedSnapshot && openCvReady) {
                            val predicted = tracker.push(bitmap, position)
                            main.post {
                                if (isCurrentFrame(token, capturedEpoch, false) && visibility == View.VISIBLE) {
                                    overlay.setTracked(predicted, frameWidth, frameHeight)
                                    lastOverlayRefreshAt = SystemClock.elapsedRealtime()
                                }
                            }
                        }
                        if (isCurrentFrame(token, capturedEpoch, pausedSnapshot) &&
                            inferenceInFlight.compareAndSet(false, true)) {
                            try {
                                worker.execute {
                                    var resultQueued = false
                                    try {
                                        if (isCurrentFrame(token, capturedEpoch, pausedSnapshot)) {
                                            val result = analyzeFrame(bitmap).copy(copyMs = copyMs)
                                            resultQueued = trackingHandler.post resultTask@{
                                                try {
                                                    if (!isCurrentFrame(token, capturedEpoch, pausedSnapshot)) return@resultTask
                                                    // Ask the existing tracker to align the source detection
                                                    // to its newest captured frame.
                                                    val corrected = if (pausedSnapshot) result.boxes else
                                                        tracker.acceptDetection(position, frameWidth, frameHeight, result.boxes)
                                                    main.post {
                                                        if (isCurrentFrame(token, capturedEpoch, pausedSnapshot) &&
                                                            visibility == View.VISIBLE) {
                                                            lastResultAgeMs = (playbackPosition() - position).coerceAtLeast(0L)
                                                            lastDetectionWasStale = corrected == null
                                                            if (pausedSnapshot) overlay.setDetection(result)
                                                            else if (corrected != null) {
                                                                overlay.setTracked(corrected, frameWidth, frameHeight)
                                                                lastOverlayRefreshAt = SystemClock.elapsedRealtime()
                                                            }
                                                            // An expired detection must not erase a more recent valid
                                                            // flow update. The existing 400 ms watchdog still applies.
                                                            val now = SystemClock.elapsedRealtime()
                                                            val fps = if (lastDetectionAt == 0L) 0.0
                                                                else 1000.0 / (now - lastDetectionAt).coerceAtLeast(1L)
                                                            lastDetectionAt = now
                                                            onDetection(result, position, fps)
                                                        }
                                                    }
                                                } catch (e: Exception) {
                                                    tracker.reset()
                                                    main.post {
                                                        if (isCurrentFrame(token, capturedEpoch, pausedSnapshot)) {
                                                            clearOverlay()
                                                            onFailure("Optical flow compensation: ${e.message}")
                                                        }
                                                    }
                                                } finally {
                                                    // Keep the single-job gate closed until flow correction is ready.
                                                    inferenceInFlight.set(false)
                                                    requestFreshFrame()
                                                }
                                            }
                                        }
                                    } catch (e: Exception) {
                                        main.post {
                                            if (isCurrentFrame(token, capturedEpoch, pausedSnapshot))
                                                onFailure("Inference: ${e.message}")
                                        }
                                    } finally {
                                        bitmap.recycle()
                                        if (!resultQueued) {
                                            inferenceInFlight.set(false)
                                            requestFreshFrame()
                                        }
                                    }
                                }
                                handedToWorker = true
                            } catch (e: Exception) {
                                inferenceInFlight.set(false)
                                main.post {
                                    if (isCurrentFrame(token, capturedEpoch, pausedSnapshot))
                                        onFailure("Inference queue: ${e.message}")
                                }
                            }
                        }
                    } catch (e: Exception) {
                        tracker.reset()
                        main.post {
                            if (isCurrentFrame(token, capturedEpoch, pausedSnapshot)) {
                                clearOverlay()
                                onFailure("Optical flow: ${e.message}")
                            }
                        }
                    } finally {
                        if (!handedToWorker) bitmap.recycle()
                        copyInFlight.set(false)
                    }
                }
                if (!posted) {
                    bitmap.recycle()
                    copyInFlight.set(false)
                }
            }, copyHandler)
        } catch (e: Exception) {
            bitmap.recycle()
            copyInFlight.set(false)
            onFailure("PixelCopy error: ${e.message}")
        }
    }

    private fun reportCopyError(status: Int) {
        main.post {
            val now = SystemClock.elapsedRealtime()
            if (now - lastCopyErrorAt > 3000L && !paused) {
                lastCopyErrorAt = now
                onFailure("PixelCopy failed ($status). Try turning Detect OFF to test playback.")
            }
        }
    }

    private fun updateAspectLayout() {
        val availableW = width
        val availableH = height
        if (availableW <= 0 || availableH <= 0 || videoWidth <= 0 || videoHeight <= 0) return
        val scale = min(availableW.toFloat() / videoWidth, availableH.toFloat() / videoHeight)
        val fittedW = (videoWidth * scale).roundToInt().coerceAtLeast(1)
        val fittedH = (videoHeight * scale).roundToInt().coerceAtLeast(1)
        if (surface.layoutParams.width == fittedW && surface.layoutParams.height == fittedH) return
        surface.layoutParams = LayoutParams(fittedW, fittedH, Gravity.CENTER)
        overlay.layoutParams = LayoutParams(fittedW, fittedH, Gravity.CENTER)
    }

    override fun surfaceCreated(holder: SurfaceHolder) {
        surfaceReady = true
        if (selectedUri != null && player == null) preparePlayer()
    }

    override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) = Unit

    override fun surfaceDestroyed(holder: SurfaceHolder) {
        surfaceReady = false
        // Surface is no longer valid; a new one triggers a fresh player.
        val uri = selectedUri
        val resumeWhenReady = !paused && !finished
        stopPlayback()
        if (uri != null) {
            selectedUri = uri
            paused = !resumeWhenReady
        }
    }

    private class DetectionOverlay(context: Context, private val className: (Int) -> String) : View(context) {
        private var boxes = emptyList<Yolo26Detector.Box>()
        private var srcW = 1
        private var srcH = 1
        private val line = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.rgb(55, 235, 128)
            style = Paint.Style.STROKE
            strokeWidth = 3f * resources.displayMetrics.density
        }
        private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.YELLOW
            textSize = 12f * resources.displayMetrics.scaledDensity
            isFakeBoldText = true
        }
        fun setDetection(d: Detection) {
            boxes = d.boxes
            srcW = d.sourceWidth.coerceAtLeast(1)
            srcH = d.sourceHeight.coerceAtLeast(1)
            invalidate()
        }
        fun setTracked(tracked: List<Yolo26Detector.Box>, sourceW: Int, sourceH: Int) {
            boxes = tracked
            srcW = sourceW.coerceAtLeast(1)
            srcH = sourceH.coerceAtLeast(1)
            invalidate()
        }
        fun clear() { boxes = emptyList(); invalidate() }
        override fun onDraw(canvas: Canvas) {
            val scaleX = width.toFloat() / srcW
            val scaleY = height.toFloat() / srcH
            for (b in boxes) {
                val x1 = b.left * scaleX
                val y1 = b.top * scaleY
                canvas.drawRect(RectF(x1, y1, b.right * scaleX, b.bottom * scaleY), line)
                canvas.drawText("${className(b.classId)}: %.2f".format(b.confidence),
                    x1, (y1 - 5f).coerceAtLeast(textPaint.textSize), textPaint)
            }
        }
    }
}