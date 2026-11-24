package com.example.weatheredgemobile

import android.content.Context

internal const val LOCAL_YOLO_SIZE = 640

class DualYoloDetector(context: Context) : AutoCloseable {
    private val onnx = Yolo26Detector(context, LOCAL_YOLO_SIZE)

    val localBackend = "ONNX"
    val comparison = ""
    val lastComparisonMs = 0.0

    internal fun prepareRestored(): Double = 0.0

    internal fun predictRestored(
        frame: RestoredFrame
    ): Yolo26Detector.Prediction {
        return onnx.predictRestored(frame)
    }

    override fun close() {
        onnx.close()
    }
}