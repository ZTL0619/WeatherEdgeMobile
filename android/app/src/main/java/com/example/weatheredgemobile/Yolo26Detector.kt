package com.example.weatheredgemobile

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import java.nio.FloatBuffer
import kotlin.math.min

/** Supports YOLO26 DETECTION, exported with nms=False, shape [1,300,6]. */
class Yolo26Detector(context: Context, private val inputSize: Int = 640) : AutoCloseable {
    data class Box(val left: Float, val top: Float, val right: Float, val bottom: Float,
                   val confidence: Float, val classId: Int)
    data class Prediction(val boxes: List<Box>, val inferenceMs: Double, val preprocessMs: Double)

    private val threshold = 0.15f
    private val environment = OrtEnvironment.getEnvironment()
    private val options = OrtSession.SessionOptions().apply { setIntraOpNumThreads(4) }
    private val session: OrtSession
    private val inputName: String
    private val floatLetterbox by lazy { FloatLetterbox(inputSize) }
    private val padding by lazy { FloatArray(inputSize * inputSize) { 114f / 255f } }

    init {
        require(inputSize in setOf(256, 384, 512, 640))
        val asset = if (inputSize == 640) "yolo26n.onnx" else "yolo26n_${inputSize}.onnx"
        val modelData = context.assets.open(asset).use { it.readBytes() }
        session = environment.createSession(modelData, options)
        inputName = session.inputNames.first()
        val inputInfo = session.inputInfo.getValue(inputName).info as ai.onnxruntime.TensorInfo
        require(inputSize == 640 || inputInfo.shape.contentEquals(longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong()))) {
            "Unexpected input shape for $asset: ${inputInfo.shape.contentToString()}"
        }
        // Fail early on a wrong export rather than draw nonsensical boxes.
        val outputInfo = session.outputInfo.values.first().info as ai.onnxruntime.TensorInfo
        val shape = outputInfo.shape
        require(shape.size == 3 && shape[1] == 300L && shape[2] == 6L) {
            "Expected YOLO26 nms=False output [1,300,6], got ${shape.contentToString()}"
        }
    }

    /** Returned coordinates are relative to the original, rotated bitmap. */
    fun predict(bitmap: Bitmap): Prediction {
        val t0 = System.nanoTime()
        val iw = bitmap.width.toFloat()
        val ih = bitmap.height.toFloat()
        val scale = min(inputSize / iw, inputSize / ih)
        val newW = iw * scale
        val newH = ih * scale
        val dx = (inputSize - newW) / 2f
        val dy = (inputSize - newH) / 2f
        val letterbox = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(letterbox)
        canvas.drawColor(Color.rgb(114, 114, 114))
        canvas.drawBitmap(bitmap, null, RectF(dx, dy, dx + newW, dy + newH),
                          Paint(Paint.FILTER_BITMAP_FLAG))
        val pixels = IntArray(inputSize * inputSize)
        letterbox.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        letterbox.recycle()
        val plane = inputSize * inputSize
        val input = FloatArray(3 * plane)
        for (i in pixels.indices) {
            val px = pixels[i]
            input[i] = Color.red(px) / 255f
            input[plane + i] = Color.green(px) / 255f
            input[2 * plane + i] = Color.blue(px) / 255f
        }
        return infer(FloatBuffer.wrap(input), iw, ih, scale, scale, dx, dy, t0)
    }

    /** At 256 share the output tensor; otherwise interpolate the valid ROI in float32.
     * Coordinates always refer to the unpadded restored ROI, independent of detector size. */
    internal fun predictRestored(frame: RestoredFrame): Prediction {
        val t0 = System.nanoTime()
        require(frame.stride == 256)
        require(frame.width > 0 && frame.height > 0 && frame.left >= 0 && frame.top >= 0 &&
            frame.left + frame.width <= frame.stride && frame.top + frame.height <= frame.stride)
        if (inputSize != frame.stride) {
            val geometry = floatLetterbox.prepare(frame)
            return infer(FloatBuffer.wrap(floatLetterbox.data),
                frame.width.toFloat(), frame.height.toFloat(),
                geometry.scaleX, geometry.scaleY, geometry.left.toFloat(), geometry.top.toFloat(), t0)
        }
        frame.restorePadding(padding)
        // Do not close this tensor here: MobileIeEnhancer owns it.
        val t1 = System.nanoTime()
        return inferTensor(frame.tensor, frame.width.toFloat(), frame.height.toFloat(),
            1f, 1f, frame.left.toFloat(), frame.top.toFloat(), t0, t1)
    }

    private fun infer(input: FloatBuffer, iw: Float, ih: Float, scaleX: Float,
                      scaleY: Float, dx: Float, dy: Float, t0: Long): Prediction {
        val t1 = System.nanoTime()
        return OnnxTensor.createTensor(environment, input,
            longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())).use { tensor ->
            inferTensor(tensor, iw, ih, scaleX, scaleY, dx, dy, t0, t1)
        }
    }

    private fun inferTensor(tensor: OnnxTensor, iw: Float, ih: Float, scaleX: Float,
                            scaleY: Float, dx: Float, dy: Float, t0: Long, t1: Long): Prediction {
        val rows = session.run(mapOf(inputName to tensor)).use { outputs ->
            @Suppress("UNCHECKED_CAST")
            (outputs[0].value as Array<Array<FloatArray>>)[0]
        }
        val t2 = System.nanoTime()
        val boxes = ArrayList<Box>()
        for (d in rows) {
            if (d.size < 6 || d[4] < threshold || !d[4].isFinite()) continue
            val left = ((d[0] - dx) / scaleX).coerceIn(0f, iw)
            val top = ((d[1] - dy) / scaleY).coerceIn(0f, ih)
            val right = ((d[2] - dx) / scaleX).coerceIn(0f, iw)
            val bottom = ((d[3] - dy) / scaleY).coerceIn(0f, ih)
            if (right <= left || bottom <= top) continue
            boxes += Box(left, top, right, bottom, d[4], d[5].toInt())
        }
        return Prediction(boxes, (t2 - t1) / 1e6, (t1 - t0) / 1e6)
    }

    override fun close() {
        session.close()
        options.close()
    }
}
