package com.example.weatheredgemobile

import kotlin.math.roundToInt

/** CPU float32 ROI resize + YOLO letterbox. Single worker owns reusable buffers. */
internal class FloatLetterbox(private val size: Int) {
    internal data class Geometry(val left: Int, val top: Int, val scaleX: Float, val scaleY: Float)
    val data = FloatArray(3 * size * size)
    private var lastWidth = -1
    private var lastHeight = -1
    private var w = 0
    private var h = 0
    private var left = 0
    private var top = 0

    fun prepare(frame: RestoredFrame): Geometry {
        require(frame.width > 0 && frame.height > 0 && frame.stride > 0 &&
            frame.left >= 0 && frame.top >= 0 && frame.left + frame.width <= frame.stride &&
            frame.top + frame.height <= frame.stride && frame.rgb.size == 3 * frame.stride * frame.stride)
        if (lastWidth != frame.width || lastHeight != frame.height) {
            val scale = minOf(size.toFloat() / frame.width, size.toFloat() / frame.height)
            w = (frame.width * scale).roundToInt().coerceIn(1, size)
            h = (frame.height * scale).roundToInt().coerceIn(1, size)
            left = (size - w) / 2
            top = (size - h) / 2
            lastWidth = frame.width
            lastHeight = frame.height
        }
        NativePreview.ensureLoaded()
        NativePreview.resizeLetterbox(frame.rgb, frame.stride, frame.left, frame.top,
            frame.width, frame.height, data, size, left, top, w, h)
        return Geometry(left, top, w.toFloat() / frame.width, h.toFloat() / frame.height)
    }
}
