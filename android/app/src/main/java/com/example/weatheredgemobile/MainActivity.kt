package com.example.weatheredgemobile

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.ImageDecoder
import android.graphics.Matrix
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.util.TypedValue
import android.view.Gravity
import android.view.ViewGroup
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import android.view.View
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * Shared MobileIE + YOLO pipeline for image, camera and sampled video frames.
 * Video keeps its independent playback and tracking on the original image coordinates.
 */
class MainActivity : ComponentActivity() {
    private enum class InputMode { CAMERA, IMAGE, VIDEO }

    private val cocoClassNames = arrayOf(
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
        "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    )

    @Volatile private var mode: InputMode = InputMode.CAMERA
    @Volatile private var imageRequestId = 0L
    private lateinit var pauseVideoButton: Button
    private lateinit var detectVideoButton: Button
    @Volatile private var backendName: String = "ONNX"
    private lateinit var videoView: ContinuousVideoView

    private lateinit var frameView: DetectionPreviewView
    private lateinit var statusView: TextView
    private lateinit var worker: ExecutorService
    private var cameraProvider: ProcessCameraProvider? = null
    private var detector: DualYoloDetector? = null // all accesses on the worker thread
    private var localEnhancer: MobileIeEnhancer? = null // owned by worker; lazy load
    private val cameraUiPending = AtomicBoolean(false)

    private val requestCamera = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (mode == InputMode.CAMERA) {
            if (granted) startCamera() else setStatus("Camera permission denied. You can still use Local Image.")
        }
    }
    private val selectImage = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) showPickedImage(uri)
    }
    private val selectVideo = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) showPickedVideo(uri)
    }


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.statusBarColor = Color.BLACK
        window.navigationBarColor = Color.BLACK
        worker = Executors.newSingleThreadExecutor()

        val root = FrameLayout(this).apply { setBackgroundColor(Color.BLACK) }
        frameView = DetectionPreviewView(this)
        root.addView(frameView, FrameLayout.LayoutParams(-1, -1))
        videoView = ContinuousVideoView(
            this,
            worker,
            analyzeFrame = { bitmap -> analyzeVideoFrame(bitmap) },
            onDetection = { detection, positionMs, fps ->
                if (mode == InputMode.VIDEO && !isDestroyed) {
                    val syncStatus = when {
                        videoView.isPaused -> "PAUSED FRAME: aligned"
                        videoView.lastDetectionWasStale ->
                            "TRACK LOST / source frame expired"
                        else -> "FLOW TRACKING | detection age ${videoView.lastResultAgeMs} ms"
                    }
                    setStatus("Video | MobileIE + YOLO26n/ONNX | ${detection.boxes.size} objects | %.1f s\n".format(positionMs / 1000.0) +
                        "copy %.1f | IE + pre %.1f | YOLO %.1f ms\n".format(
                            detection.copyMs, detection.preprocessMs, detection.inferenceMs) +
                        "Detection FPS %.1f | playback x%.2f | %s".format(fps, videoView.playbackRate, syncStatus))
                }
            },
            onPlaybackDone = {
                if (mode == InputMode.VIDEO && !isDestroyed) {
                    pauseVideoButton.text = "Replay Video"
                    setStatus("Video finished. Tap Replay Video or select another file.")
                }
            },
            onPlaybackHealth = { positionMs, speed ->
                if (mode == InputMode.VIDEO && !isDestroyed && !videoView.isDetectionEnabled) {
                    setStatus("Playback ONLY | %.1f s | speed x%.2f\n".format(positionMs / 1000.0, speed) +
                        "Detection OFF: video decoder and renderer test")
                }
            },
            onFailure = { message ->
                if (mode == InputMode.VIDEO && !isDestroyed) setStatus("Video error: $message")
            },
            className = { id -> cocoClassNames.getOrNull(id) ?: "ID $id" }
        )
        videoView.visibility = View.GONE
        root.addView(videoView, FrameLayout.LayoutParams(-1, -1))

        statusView = TextView(this).apply {
            setBackgroundColor(0xBB000000.toInt())
            setTextColor(Color.WHITE)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f)
            setPadding(dp(12), dp(9), dp(12), dp(9))
            text = "Loading YOLO26n ..."
        }
        root.addView(statusView, FrameLayout.LayoutParams(-1, -2, Gravity.TOP))

        // Small three-mode toolbar, with Pause/Resume visible only in video mode.
        val controls = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(0xAA000000.toInt())
        }
        val tools = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            setPadding(dp(4), dp(2), dp(4), dp(2))
        }
        fun addModeButton(title: String, action: () -> Unit) {
            val button = Button(this).apply {
                text = title
                isAllCaps = false
                minWidth = 0
                minimumWidth = 0
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 11f)
                setPadding(0, 0, 0, 0)
                setOnClickListener { action() }
            }
            tools.addView(button, LinearLayout.LayoutParams(0, dp(46), 1f))
        }
        addModeButton("Live Camera") {
            mode = InputMode.CAMERA
            imageRequestId++            // Cancels any stale image jobs.
            videoView.stopPlayback()
            videoView.visibility = View.GONE
            frameView.visibility = View.VISIBLE
            pauseVideoButton.visibility = View.GONE
            detectVideoButton.visibility = View.GONE
            setStatus("Starting camera ...")
            checkCameraPermission()
        }
        addModeButton("Local Image") {
            if (mode == InputMode.VIDEO) videoView.pausePlayback()
            selectImage.launch("image/*")
        }
        addModeButton("Local Video") {
            if (mode == InputMode.VIDEO) videoView.pausePlayback()
            selectVideo.launch("video/*")
        }
        controls.addView(tools)

        pauseVideoButton = Button(this).apply {
            text = "Pause Video"
            isAllCaps = false
            visibility = View.GONE
            setOnClickListener {
                if (mode != InputMode.VIDEO) return@setOnClickListener
                if (videoView.isFinished) {
                    videoView.replay()
                    text = "Pause Video"
                } else if (videoView.isPaused) {
                    videoView.resumePlayback()
                    text = "Pause Video"
                } else {
                    videoView.pausePlayback(inspectFrame = true)
                    text = "Resume Video"
                }
            }
        }
        controls.addView(pauseVideoButton, LinearLayout.LayoutParams(-1, dp(44)))
        detectVideoButton = Button(this).apply {
            text = "Detection ON (tap to compare playback)"
            isAllCaps = false
            visibility = View.GONE
            setOnClickListener {
                if (mode != InputMode.VIDEO) return@setOnClickListener
                val enable = !videoView.isDetectionEnabled
                videoView.setDetectionEnabled(enable)
                text = if (enable) "Detection ON (tap to compare playback)" else "Detection OFF (playback only)"
                if (!enable) setStatus("Video playback only — detection disabled for performance test")
            }
        }
        controls.addView(detectVideoButton, LinearLayout.LayoutParams(-1, dp(42)))

        root.addView(controls, FrameLayout.LayoutParams(-1, -2, Gravity.BOTTOM))

        // Keep buttons/text clear of Android 15+ edge-to-edge system bars.
        ViewCompat.setOnApplyWindowInsetsListener(root) { _, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            (statusView.layoutParams as FrameLayout.LayoutParams).apply {
                topMargin = bars.top
                statusView.layoutParams = this
            }
            (controls.layoutParams as FrameLayout.LayoutParams).apply {
                bottomMargin = bars.bottom
                controls.layoutParams = this
            }
            insets
        }
        setContentView(root)

        // A single worker owns ONNX/NCNN inference: camera and image never call the session concurrently.
        worker.execute {
            try {
                detector = DualYoloDetector(this)
                runOnUiThread {
                    if (!isDestroyed && mode == InputMode.CAMERA) checkCameraPermission()
                }
            } catch (e: Exception) {
                runOnUiThread {
                    if (!isDestroyed) {
                        setStatus("Model load failed: ${e.message}\nCheck assets/yolo26n.onnx")
                        if (mode == InputMode.CAMERA) checkCameraPermission()
                    }
                }
            }
        }
    }

    private fun dp(value: Int): Int = (resources.displayMetrics.density * value).roundToInt()
    private fun setStatus(value: String) { statusView.text = value }

    private fun checkCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            requestCamera.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        if (mode != InputMode.CAMERA) return
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            if (mode != InputMode.CAMERA || isDestroyed) return@addListener
            try {
                val provider = future.get()
                cameraProvider = provider
                val analysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .build()
                analysis.setAnalyzer(worker) { image ->
                    val requestId = imageRequestId
                    // Never queue a backlog of processed previews on the UI thread.
                    if (mode != InputMode.CAMERA || !cameraUiPending.compareAndSet(false, true)) {
                        image.close()
                        return@setAnalyzer
                    }
                    var posted = false
                    try {
                        val source = image.toBitmap()
                        val result = try {
                            val degrees = image.imageInfo.rotationDegrees
                            val frame = if (degrees != 0) {
                                Bitmap.createBitmap(source, 0, 0, source.width, source.height,
                                    Matrix().apply { postRotate(degrees.toFloat()) }, true)
                            } else source
                            try { runPipeline(frame) } finally {
                                if (frame !== source && !frame.isRecycled) frame.recycle()
                            }
                        } finally {
                            if (!source.isRecycled) source.recycle()
                        }
                        runOnUiThread {
                            try {
                                if (!isDestroyed && mode == InputMode.CAMERA && requestId == imageRequestId) {
                                    showResult(result, "Camera")
                                } else {
                                    result.bitmap.recycle()
                                }
                            } finally { cameraUiPending.set(false) }
                        }
                        posted = true
                    } catch (e: Exception) {
                        runOnUiThread {
                            if (!isDestroyed && mode == InputMode.CAMERA && requestId == imageRequestId)
                                setStatus("Camera inference error: ${e.message}")
                        }
                    } finally {
                        image.close()
                        if (!posted) cameraUiPending.set(false)
                    }
                }
                provider.unbindAll()
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, analysis)
            } catch (e: Exception) {
                setStatus("Cannot start camera: ${e.message}")
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun showPickedImage(uri: Uri) {
        mode = InputMode.IMAGE
        videoView.stopPlayback()
        videoView.visibility = View.GONE
        frameView.visibility = View.VISIBLE
        pauseVideoButton.visibility = View.GONE
        detectVideoButton.visibility = View.GONE
        val requestId = ++imageRequestId
        cameraProvider?.unbindAll()
        setStatus("Loading local image ...")
        worker.execute {
            try {
                val bitmap = decodeImage(uri)
                if (mode != InputMode.IMAGE || requestId != imageRequestId) {
                    bitmap.recycle()
                    return@execute
                }
                val result = try { runPipeline(bitmap) } finally { bitmap.recycle() }
                runOnUiThread {
                    if (!isDestroyed && mode == InputMode.IMAGE && requestId == imageRequestId) {
                        showResult(result, "Image")
                    } else {
                        result.bitmap.recycle()
                    }
                }
            } catch (e: Exception) {
                runOnUiThread {
                    if (!isDestroyed && mode == InputMode.IMAGE && requestId == imageRequestId) {
                        setStatus("Cannot process image: ${e.message}")
                    }
                }
            }
        }
    }

    /** Continuous decoder and video playback; inference samples fresh frames asynchronously. */
    private fun showPickedVideo(uri: Uri) {
        mode = InputMode.VIDEO
        ++imageRequestId
        cameraProvider?.unbindAll()
        frameView.clear()
        frameView.visibility = View.GONE
        videoView.visibility = View.VISIBLE
        pauseVideoButton.text = "Pause Video"
        pauseVideoButton.visibility = View.VISIBLE
        detectVideoButton.visibility = View.VISIBLE
        videoView.setDetectionEnabled(true)
        detectVideoButton.text = "Detection ON (tap to compare playback)"
        setStatus("Opening video ...")
        videoView.play(uri)
    }

    /** Video detects on restored frames, but displays/tracks on the original video. */
    private fun analyzeVideoFrame(frame: Bitmap): ContinuousVideoView.Detection {
        val student = getStudent()
        detector?.prepareRestored()
        val started = System.nanoTime()
        val restored = student.enhanceTensor(frame)
        val prediction = detector?.predictRestored(restored)
        val comparisonMs = detector?.lastComparisonMs ?: 0.0
        // YOLO boxes refer to the unpadded restored ROI. Map back to the exact
        // source dimensions expected by the existing optical flow tracker.
        val sx = frame.width.toFloat() / restored.width
        val sy = frame.height.toFloat() / restored.height
        val boxes = prediction?.boxes.orEmpty().map { box ->
            box.copy(left = box.left * sx, top = box.top * sy,
                right = box.right * sx, bottom = box.bottom * sy)
        }
        return ContinuousVideoView.Detection(
            boxes = boxes,
            sourceWidth = frame.width,
            sourceHeight = frame.height,
            preprocessMs = student.preprocessMs + student.inferenceMs + student.postprocessMs +
                (prediction?.preprocessMs ?: 0.0),
            inferenceMs = prediction?.inferenceMs ?: 0.0,
            pipelineMs = ((System.nanoTime() - started) / 1e6 - comparisonMs).coerceAtLeast(0.0)
        )
    }

    private fun getStudent(): MobileIeEnhancer = localEnhancer ?: MobileIeEnhancer(this).also {
        localEnhancer = it
    }

    /** Bounded decoding avoids OOM on high-resolution camera/gallery photos. */
    private fun decodeImage(uri: Uri): Bitmap {
        val maxEdge = 1600
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            val source = ImageDecoder.createSource(contentResolver, uri)
            return ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
                decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                val w = info.size.width
                val h = info.size.height
                val larger = max(w, h)
                if (larger > maxEdge) {
                    val factor = maxEdge.toFloat() / larger
                    decoder.setTargetSize((w * factor).roundToInt().coerceAtLeast(1),
                        (h * factor).roundToInt().coerceAtLeast(1))
                }
            }
        }
        // Android 8.x fallback (API 26-27).
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        val boundsStream = contentResolver.openInputStream(uri)
            ?: error("Unable to open selected image stream")
        boundsStream.use { BitmapFactory.decodeStream(it, null, bounds) }
        require(bounds.outWidth > 0 && bounds.outHeight > 0) { "Unsupported image" }
        var sample = 1
        while (max(bounds.outWidth, bounds.outHeight) / sample > maxEdge) sample *= 2
        val options = BitmapFactory.Options().apply {
            inSampleSize = sample
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        return contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, options) }
            ?: error("Unable to decode image")
    }

    private data class Result(
        val bitmap: Bitmap,
        val count: Int,
        val preprocessingMs: Double,
        val inferenceMs: Double,
        val pipelineMs: Double,
        val modelLoaded: Boolean,
        val enhanced: Boolean = false,
        val enhancementMs: Double = 0.0,
        val enhancementInferenceMs: Double = 0.0,
        val enhancementLoadMs: Double = 0.0,
        val enhancementPrepMs: Double = 0.0,
        val enhancementPostMs: Double = 0.0,
        val enhancementWarmMeanMs: Double = 0.0,
        val previewMs: Double = 0.0,
        val previewReadMs: Double = 0.0,
        val previewPixelMs: Double = 0.0,
        val previewBitmapMs: Double = 0.0,
        val overlays: List<PreviewOverlay> = emptyList(),
        val readyNanos: Long = 0L,
        val localBackend: String = "ONNX",
        val comparison: String = "",
        val comparisonMs: Double = 0.0
    )

    /** Image and camera use exactly the same restoration/detection/preview pipeline. */
    private fun runPipeline(frame: Bitmap): Result {
        val loadStart = System.nanoTime()
        val alreadyLoaded = localEnhancer != null
        val student = getStudent()
        var loadMs = if (alreadyLoaded) 0.0 else (System.nanoTime() - loadStart) / 1e6
        loadMs += detector?.prepareRestored() ?: 0.0
        val start = System.nanoTime()
        val restored = student.enhanceTensor(frame)
        val enhanceMs = (System.nanoTime() - start) / 1e6
        val prediction = detector?.predictRestored(restored)
        val comparisonMs = detector?.lastComparisonMs ?: 0.0
        // Only build a display bitmap AFTER both models have finished.
        val previewStart = System.nanoTime()
        val preview = student.makePreview(restored)
        val overlays = prediction?.boxes.orEmpty().map { box ->
            val name = cocoClassNames.getOrNull(box.classId) ?: "ID ${box.classId}"
            PreviewOverlay(box.left, box.top, box.right, box.bottom,
                "$name: ${previewConfidence(box.confidence)}")
        }
        val readyNanos = System.nanoTime()
        val previewMs = (readyNanos - previewStart) / 1e6
        return Result(
            bitmap = preview,
            count = prediction?.boxes?.size ?: 0,
            preprocessingMs = prediction?.preprocessMs ?: 0.0,
            inferenceMs = prediction?.inferenceMs ?: 0.0,
            pipelineMs = ((readyNanos - start) / 1e6 - comparisonMs).coerceAtLeast(0.0),
            modelLoaded = prediction != null,
            enhanced = true,
            enhancementMs = enhanceMs,
            enhancementInferenceMs = student.inferenceMs,
            enhancementLoadMs = loadMs,
            enhancementPrepMs = student.preprocessMs,
            enhancementPostMs = student.postprocessMs,
            enhancementWarmMeanMs = student.warmMeanMs,
            previewMs = previewMs,
            previewReadMs = student.previewReadMs,
            previewPixelMs = student.previewPixelMs,
            previewBitmapMs = student.previewBitmapMs,
            overlays = overlays,
            readyNanos = readyNanos,
            localBackend = detector?.localBackend ?: "ONNX",
            comparison = detector?.comparison ?: "",
            comparisonMs = comparisonMs
        )
    }

    private fun showResult(result: Result, source: String) {
        if (result.enhanced) {
            val requestId = imageRequestId
            val expectedMode = mode
            val heading = if (result.modelLoaded) "$source | MobileIE + YOLO26n/${result.localBackend} | ${result.count} objects"
                else "$source | YOLO not loaded (image preview only)"
            val prefix = heading + "\n" +
                "IE prep %.1f | model %.1f | release %.1f ms\n".format(
                    result.enhancementPrepMs, result.enhancementInferenceMs, result.enhancementPostMs) +
                "IE CPU x4 | warm mean (20) %.1f ms\n".format(result.enhancementWarmMeanMs) +
                "ROI ${result.bitmap.width}x${result.bitmap.height} | ${if (LOCAL_YOLO_SIZE == 256) "shared tensor" else "float32 resize"} -> YOLO $LOCAL_YOLO_SIZE\n" +
                "YOLO prep %.1f | model %.1f ms\n".format(result.preprocessingMs, result.inferenceMs) +
                "Preview read %.1f | RGB->ARGB %.1f | bitmap %.1f ms\n".format(
                    result.previewReadMs, result.previewPixelMs, result.previewBitmapMs) +
                "Preview pack %.1f | worker %.1f ms\n".format(result.previewMs, result.pipelineMs)
            val load = (if (result.enhancementLoadMs > 0.0)
                "\nInit/switch + warmup %.1f ms".format(result.enhancementLoadMs) else "") +
                (if (result.comparison.isNotEmpty()) "\n" + result.comparison else "") +
                (if (result.comparisonMs > 0.0) "\nOne-time A/B %.1f ms (excluded)".format(result.comparisonMs) else "")
            setStatus(prefix + "Waiting for first draw" + load)
            frameView.submit(result.bitmap, result.overlays) { drawMs, drawEnd, hardware ->
                if (!isDestroyed && mode == expectedMode && requestId == imageRequestId) {
                    val readyToDraw = ((drawEnd - result.readyNanos) / 1e6).coerceAtLeast(0.0)
                    val toDraw = result.pipelineMs + readyToDraw
                    setStatus(prefix +
                        "UI draw CPU %.1f | to draw %.1f ms\n".format(drawMs, toDraw) +
                        "Canvas ${if (hardware) "HW" else "SW"} | ready->draw %.1f ms".format(readyToDraw) + load)
                }
            }
        } else {
            // Fallback preview path.
            frameView.submit(result.bitmap)
            if (!result.modelLoaded) {
                setStatus("$source | YOLO not loaded (image preview only)")
            } else {
                setStatus("$source | YOLO26n/$backendName | ${result.count} objects\n" +
                    "preprocess %.1f ms | inference %.1f ms | pipeline %.1f ms".format(
                        result.preprocessingMs, result.inferenceMs, result.pipelineMs))
            }
        }
    }

    override fun onStop() {
        if (mode == InputMode.VIDEO && ::videoView.isInitialized && !videoView.isPaused && !videoView.isFinished) {
            videoView.pausePlayback()
            pauseVideoButton.text = "Resume Video"
        }
        super.onStop()
    }

    override fun onDestroy() {
        if (::frameView.isInitialized) frameView.clear()
        if (::videoView.isInitialized) videoView.dispose()
        cameraProvider?.unbindAll()
        worker.execute {
            try {
                localEnhancer?.close()
            } finally {
                detector?.close()
            }
        }
        worker.shutdown()
        super.onDestroy()
    }
}
