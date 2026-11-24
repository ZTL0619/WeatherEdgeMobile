package com.example.weatheredgemobile

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.roundToInt

/** Borrowed CPU tensor. Valid until the next enhanceTensor() or close().
 * The existing exporter clamps restored to [0,1] inside the ONNX graph.
 * Copy to the reusable RGB array only when a consumer actually needs pixels. */
internal class RestoredFrame(
    val tensor: OnnxTensor, private val buffer: FloatBuffer,
    private val pixels: FloatArray, val stride: Int, val left: Int, val top: Int,
    val width: Int, val height: Int
) {
    private var read = false
    val rgb: FloatArray
        get() {
            if (!read) {
                val src = buffer.duplicate()
                src.clear()
                src.get(pixels)
                read = true
            }
            return pixels
        }

    /** Existing YOLO padding semantics, without copying the valid ROI.
     * Only the worker may touch these buffers. */
    fun restorePadding(pad: FloatArray) {
        val plane = stride * stride
        require(pad.size >= plane)
        val dst = buffer.duplicate()
        fun fill(start: Int, count: Int) {
            if (count == 0) return
            dst.position(start)
            dst.put(pad, 0, count)
        }
        for (c in 0..2) {
            val base = c * plane
            fill(base, top * stride)
            fill(base + (top + height) * stride, (stride - top - height) * stride)
            for (y in top until top + height) {
                val row = base + y * stride
                fill(row, left)
                fill(row + left + width, stride - left - width)
            }
        }
        read = false
    }
}

interface EnhancementEngine {
    fun enhance(frame: Bitmap): Bitmap
}

class IdentityEnhancer : EnhancementEngine {
    override fun enhance(frame: Bitmap): Bitmap = frame
}

/** Single-worker, single-input deterministic student. No noise or diffusion on device. */
internal class MobileIeEnhancer(context: Context) : EnhancementEngine, AutoCloseable {
    private val size = 256
    private val plane = size * size
    private val shape = longArrayOf(1, 3, 256, 256)
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val previewBitmap = PreviewBitmap()
    private val pixels = IntArray(plane)
    private val rgb = FloatArray(3 * plane)
    private val restored = FloatArray(3 * plane)
    private val output = ByteBuffer.allocateDirect(3 * plane * Float.SIZE_BYTES)
        .order(ByteOrder.nativeOrder()).asFloatBuffer()
    private val outputTensor: OnnxTensor
    private val outputFeeds: Map<String, OnnxTensor>
    private val input = ByteBuffer.allocateDirect(3 * plane * Float.SIZE_BYTES)
        .order(ByteOrder.nativeOrder()).asFloatBuffer()
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    var inferenceMs: Double = 0.0
        private set

    var preprocessMs: Double = 0.0
        private set
    var postprocessMs: Double = 0.0
        private set
    var warmMeanMs: Double = 0.0
        private set

    init {
        val bytes = context.assets.open("student_256.onnx").use { it.readBytes() }
        val options = OrtSession.SessionOptions()
        val created = try {
            options.setIntraOpNumThreads(4)
            options.setInterOpNumThreads(1)
            // Keep ORT default spinning, matching the successful isolation test.
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            env.createSession(bytes, options)
        } finally {
            options.close()
        }
        try {
            require(created.inputNames == setOf("degraded")) {
                "Expected single-input student_256.onnx; got ${created.inputNames}"
            }
            require(created.outputNames == setOf("restored")) {
                "Expected restored output; got ${created.outputNames}"
            }
        } catch (e: Exception) {
            created.close()
            throw e
        }
        session = created
        outputTensor = try {
            val info = session.outputInfo.getValue("restored").info as ai.onnxruntime.TensorInfo
            require(info.shape.contentEquals(shape) && info.type == ai.onnxruntime.OnnxJavaType.FLOAT) {
                "Unexpected restored tensor shape/type"
            }
            OnnxTensor.createTensor(env, output, shape)
        } catch (e: Exception) {
            session.close()
            throw e
        }
        outputFeeds = mapOf("restored" to outputTensor)
        // One-time diagnostic on a neutral image, with YOLO already initialized.
        // Called during lazy initialization, not included in per-image pipeline time.
        try {
            rgb.fill(0.5f)
            input.clear()
            input.put(rgb)
            input.rewind()
            OnnxTensor.createTensor(env, input, shape).use { tensor ->
                val feeds = mapOf("degraded" to tensor)
                repeat(10) { session.run(feeds, outputFeeds).close() }
                var total = 0.0
                repeat(20) {
                    val start = System.nanoTime()
                    session.run(feeds, outputFeeds).use { _ -> total += (System.nanoTime() - start) / 1e6 }
                }
                warmMeanMs = total / 20.0
            }
        } catch (e: Exception) {
            outputTensor.close()
            session.close()
            throw e
        }
    }

    /** No output bitmap is constructed here; keep the float32 NCHW data for YOLO. */
    fun enhanceTensor(frame: Bitmap): RestoredFrame {
        val prepStart = System.nanoTime()
        val scale = minOf(size.toDouble() / frame.width, size.toDouble() / frame.height)
        val w = (frame.width * scale).roundToInt().coerceIn(1, size)
        val h = (frame.height * scale).roundToInt().coerceIn(1, size)
        val left = (size - w) / 2
        val top = (size - h) / 2
        val roi = Rect(left, top, left + w, top + h)
        val square = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        try {
            Canvas(square).apply {
                drawColor(Color.rgb(114, 114, 114))
                drawBitmap(frame, null, roi, paint)
            }
            square.getPixels(pixels, 0, size, 0, 0, size, size)
        } finally {
            square.recycle()
        }
        val inv = 1f / 255f
        for (i in 0 until plane) {
            val c = pixels[i]
            rgb[i] = ((c ushr 16) and 255) * inv
            rgb[i + plane] = ((c ushr 8) and 255) * inv
            rgb[i + 2 * plane] = (c and 255) * inv
        }
        input.clear()
        input.put(rgb)
        input.rewind()
        OnnxTensor.createTensor(env, input, shape).use { tensor ->
            preprocessMs = (System.nanoTime() - prepStart) / 1e6
            val started = System.nanoTime()
            session.run(mapOf("degraded" to tensor), outputFeeds).use { _ ->
                inferenceMs = (System.nanoTime() - started) / 1e6
            }
        }
        // No getFloatBuffer() copy and no redundant clamp: export already clips.
        postprocessMs = ((System.nanoTime() - prepStart) / 1e6 - preprocessMs - inferenceMs).coerceAtLeast(0.0)
        return RestoredFrame(outputTensor, output, restored, size, left, top, w, h)
    }

    val previewReadMs: Double get() = previewBitmap.readMs
    val previewPixelMs: Double get() = previewBitmap.pixelMs
    val previewBitmapMs: Double get() = previewBitmap.bitmapMs

    /** For display only, after YOLO. The detector never reads this bitmap. */
    fun makePreview(frame: RestoredFrame): Bitmap = previewBitmap.fromRgb(frame)

    override fun enhance(frame: Bitmap): Bitmap = makePreview(enhanceTensor(frame))

    override fun close() {
        try { outputTensor.close() } finally { session.close() }
    }
}
