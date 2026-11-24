package com.example.weatheredgemobile

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Typeface
import android.view.View
import kotlin.math.roundToInt

/** Model-independent boxes in the pixel coordinates of the displayed bitmap. */
internal data class PreviewOverlay(
    val left: Float, val top: Float, val right: Float, val bottom: Float,
    val label: String
)

/** Worker-owned scratch buffer. Each returned bitmap owns independent pixels. */
internal class PreviewBitmap {
    init { NativePreview.ensureLoaded() }
    private var argb = IntArray(0)
    var readMs = 0.0
        private set
    var pixelMs = 0.0
        private set
    var bitmapMs = 0.0
        private set

    fun fromRgb(frame: RestoredFrame): Bitmap {
        val stride = frame.stride
        val plane = stride * stride
        require(frame.width > 0 && frame.height > 0 && frame.left >= 0 && frame.top >= 0 &&
            frame.left + frame.width <= stride && frame.top + frame.height <= stride)
        val area = frame.width * frame.height
        if (argb.size < area) argb = IntArray(area)
        val readStart = System.nanoTime()
        val rgb = frame.rgb // Single bulk copy, after YOLO; graph output is clamped.
        readMs = (System.nanoTime() - readStart) / 1e6
        require(rgb.size == 3 * plane)
        val pixelStart = System.nanoTime()
        // One JNI call for the entire ROI; includes any JNI array marshalling.
        NativePreview.packRgb(rgb, stride, frame.left, frame.top,
            frame.width, frame.height, argb)
        // No second bitmap, setPixels call, software drawing, or full-size upscale.
        val bitmapStart = System.nanoTime()
        pixelMs = (bitmapStart - pixelStart) / 1e6
        val bitmap = Bitmap.createBitmap(argb, frame.width, frame.height, Bitmap.Config.ARGB_8888)
        bitmapMs = (System.nanoTime() - bitmapStart) / 1e6
        return bitmap
    }
}

/** UI-thread renderer. Draw directly to the window canvas; hardware accelerated when available. */
internal class DetectionPreviewView(context: Context) : View(context) {
    private var bitmap: Bitmap? = null
    private var overlays: List<PreviewOverlay> = emptyList()
    private val destination = RectF()
    private val bitmapPaint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val stroke = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(55, 235, 128)
        style = Paint.Style.STROKE
        strokeWidth = 1.5f
    }
    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.YELLOW
        textSize = 10f
        typeface = Typeface.DEFAULT_BOLD
    }
    private var generation = 0L
    private var firstDraw: ((Double, Long, Boolean) -> Unit)? = null

    /** Ownership transfers to the view. Never recycle a displayed bitmap manually:
     * the hardware render thread may still hold it. The GC releases replaced images. */
    fun submit(image: Bitmap, boxes: List<PreviewOverlay> = emptyList(),
               onFirstDraw: ((Double, Long, Boolean) -> Unit)? = null) {
        bitmap = image
        overlays = boxes
        generation++
        firstDraw = onFirstDraw
        invalidate()
    }

    fun clear() {
        generation++
        firstDraw = null
        bitmap = null
        overlays = emptyList()
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val image = bitmap ?: return
        if (width <= 0 || height <= 0 || image.isRecycled) return
        val started = System.nanoTime()
        val scale = minOf(width.toFloat() / image.width, height.toFloat() / image.height)
        val left = (width - image.width * scale) * 0.5f
        val top = (height - image.height * scale) * 0.5f
        destination.set(left, top, left + image.width * scale, top + image.height * scale)
        canvas.drawBitmap(image, null, destination, bitmapPaint)
        if (overlays.isNotEmpty()) {
            val saved = canvas.save()
            canvas.clipRect(destination)
            canvas.translate(left, top)
            canvas.scale(scale, scale)
            for (box in overlays) {
                canvas.drawRect(box.left, box.top, box.right, box.bottom, stroke)
                canvas.drawText(box.label, box.left, (box.top - 2f).coerceAtLeast(10f), labelPaint)
            }
            canvas.restoreToCount(saved)
        }
        val ended = System.nanoTime()
        val callback = firstDraw
        if (callback != null) {
            firstDraw = null
            val token = generation
            val hardware = canvas.isHardwareAccelerated
            // Report on the UI thread after onDraw, not in the draw traversal.
            post { if (generation == token) callback((ended - started) / 1e6, ended, hardware) }
        }
    }
}

/** Avoid constructing a Java Formatter for every detection. */
internal fun previewConfidence(value: Float): String {
    val n = (value.coerceIn(0f, 1f) * 100f).roundToInt()
    val frac = n % 100
    return "${n / 100}.${if (frac < 10) "0" else ""}$frac"
}
