package com.example.weatheredgemobile

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.video.Video
import java.util.ArrayDeque
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Lightweight frame-to-frame optical flow for the *video overlay only*.
 * v10: forward-backward LK + RANSAC partial-affine fit for translation and scale.
 * Confined to a single handler thread; callers must not access from other threads.
 * Stores a short history so slow YOLO detections can be transported from the
 * original video frame to the latest available frame.
 *
 * This is NOT a multi-object re-identification tracker. Lost tracks expire.
 */
internal class SparseFlowTracker {
    private data class Frame(val videoMs: Long, val gray: Mat, val width: Int, val height: Int)
    private data class Track(
        var left: Float, var top: Float, var right: Float, var bottom: Float,
        val confidence: Float, val classId: Int, val detectionVideoMs: Long
    )

    private val history = ArrayDeque<Frame>()
    private val tracks = ArrayList<Track>()
    private val maxHistoryMs = 2500L
    private val maxTrackAgeMs = 1800L
    private val maxFrames = 24
    private val workingEdge = 480
    private val maxBoxes = 35
    private var originalW = 1
    private var originalH = 1

    fun reset() {
        history.forEach { it.gray.release() }
        history.clear()
        tracks.clear()
    }

    fun push(bitmap: Bitmap, videoMs: Long): List<Yolo26Detector.Box> {
        val mat = Mat()
        val gray = Mat()
        val small = Mat()
        try {
            Utils.bitmapToMat(bitmap, mat)
            Imgproc.cvtColor(mat, gray, Imgproc.COLOR_RGBA2GRAY)
            val ratio = min(1.0, workingEdge.toDouble() / max(bitmap.width, bitmap.height))
            val w = max(1, (bitmap.width * ratio).toInt())
            val h = max(1, (bitmap.height * ratio).toInt())
            Imgproc.resize(gray, small, Size(w.toDouble(), h.toDouble()))
            originalW = bitmap.width
            originalH = bitmap.height
        } finally {
            mat.release()
            gray.release()
        }
        val current = Frame(videoMs, small, small.cols(), small.rows())
        val last = history.peekLast()
        if (last != null && last.width == current.width && last.height == current.height) {
            val iterator = tracks.iterator()
            while (iterator.hasNext()) {
                val track = iterator.next()
                if (videoMs - track.detectionVideoMs > maxTrackAgeMs ||
                    !advance(track, last.gray, current.gray)) iterator.remove()
            }
        } else {
            tracks.clear()
        }
        history.addLast(current)
        while (history.size > maxFrames ||
            (history.size > 2 && videoMs - history.peekFirst().videoMs > maxHistoryMs)) {
            history.removeFirst().gray.release()
        }
        return boxes()
    }

    /** Returns null when the detection's source frame fell out of history. */
    fun acceptDetection(
        sourceVideoMs: Long, sourceW: Int, sourceH: Int,
        detections: List<Yolo26Detector.Box>
    ): List<Yolo26Detector.Box>? {
        if (history.isEmpty() || sourceW <= 0 || sourceH <= 0) return null
        val frames = history.toList()
        var anchor = 0
        var bestDelta = Long.MAX_VALUE
        for (i in frames.indices) {
            val delta = abs(frames[i].videoMs - sourceVideoMs)
            if (delta < bestDelta) { bestDelta = delta; anchor = i }
        }
        // PixelCopy/MediaPlayer positions are approximate, not decoder PTS.
        if (bestDelta > 260L || frames.last().videoMs - sourceVideoMs > maxHistoryMs) return null
        val ref = frames[anchor]
        val factorX = ref.width.toFloat() / sourceW
        val factorY = ref.height.toFloat() / sourceH
        val incoming = detections.asSequence().filter { it.confidence.isFinite() }
            .sortedByDescending { it.confidence }.take(maxBoxes).map { b ->
                Track(b.left * factorX, b.top * factorY,
                    b.right * factorX, b.bottom * factorY,
                    b.confidence, b.classId, sourceVideoMs)
            }.toMutableList()
        for (j in anchor until frames.lastIndex) {
            val a = frames[j]
            val b = frames[j + 1]
            val it = incoming.iterator()
            while (it.hasNext()) if (!advance(it.next(), a.gray, b.gray)) it.remove()
        }
        // Reinitialize from newest detector snapshot, transported by optical flow.
        tracks.clear()
        tracks.addAll(incoming)
        return boxes()
    }

    private fun boxes(): List<Yolo26Detector.Box> {
        val current = history.peekLast() ?: return emptyList()
        val sx = originalW.toFloat() / current.width
        val sy = originalH.toFloat() / current.height
        return tracks.map { b ->
            Yolo26Detector.Box(b.left * sx, b.top * sy, b.right * sx, b.bottom * sy,
                b.confidence, b.classId)
        }
    }

    /**
     * Track vehicle centre AND its apparent size.  Only features with a
     * forward/backward LK-consistent match are allowed into the robust fit.
     * A partial affine transform gives translation + uniform scale, while
     * RANSAC suppresses road/background outliers.
     *
     * If too few spread-out inliers exist to estimate scale, fall back to a
     * robust translation, rather than inventing a scale change. A track with
     * no reliable motion is removed instead of freezing the old rectangle.
     */
    private fun advance(t: Track, prev: Mat, next: Mat): Boolean {
        val x1 = t.left.toInt().coerceIn(0, prev.cols() - 1)
        val y1 = t.top.toInt().coerceIn(0, prev.rows() - 1)
        val x2 = t.right.toInt().coerceIn(0, prev.cols())
        val y2 = t.bottom.toInt().coerceIn(0, prev.rows())
        val w = x2 - x1
        val h = y2 - y1
        if (w < 8 || h < 8) return false

        // Keep the features on the visible interior, not the surrounding road.
        val mx = max(1, (w * 0.16f).toInt())
        val my = max(1, (h * 0.16f).toInt())
        val coreRect = Rect(x1 + mx, y1 + my, w - 2 * mx, h - 2 * my)
        if (coreRect.width < 5 || coreRect.height < 5) return false

        val regionPrev = prev.submat(coreRect)
        val regionNext = next.submat(coreRect)
        val diff = Mat()
        val corners = MatOfPoint()
        val p0 = MatOfPoint2f()
        val p1 = MatOfPoint2f()
        val pBack = MatOfPoint2f()
        val statusF = MatOfByte()
        val statusB = MatOfByte()
        val errorF = MatOfFloat()
        val errorB = MatOfFloat()
        val affineInliers = Mat()
        val fittedFrom = MatOfPoint2f()
        val fittedTo = MatOfPoint2f()
        var transform: Mat? = null
        try {
            Core.absdiff(regionPrev, regionNext, diff)
            val appearanceChange = Core.mean(diff).`val`[0]
            Imgproc.goodFeaturesToTrack(regionPrev, corners, 48, 0.018, 2.0)
            val found = corners.toArray()

            var translate: Pair<Float, Float>? = null
            var scale = 1f
            if (found.size >= 4) {
                p0.fromArray(*found.map { Point(it.x + coreRect.x, it.y + coreRect.y) }.toTypedArray())
                Video.calcOpticalFlowPyrLK(prev, next, p0, p1, statusF, errorF)
                Video.calcOpticalFlowPyrLK(next, prev, p1, pBack, statusB, errorB)
                val pts0 = p0.toArray()
                val pts1 = p1.toArray()
                val ptsBack = pBack.toArray()
                val fStatus = statusF.toArray()
                val bStatus = statusB.toArray()
                val validFrom = ArrayList<Point>()
                val validTo = ArrayList<Point>()
                val shiftsX = ArrayList<Double>()
                val shiftsY = ArrayList<Double>()

                for (i in 0 until min(pts0.size, min(pts1.size, min(ptsBack.size, min(fStatus.size, bStatus.size))))) {
                    if (fStatus[i].toInt() == 0 || bStatus[i].toInt() == 0) continue
                    val src = pts0[i]
                    val dst = pts1[i]
                    val back = ptsBack[i]
                    val dx = dst.x - src.x
                    val dy = dst.y - src.y
                    val fbError = kotlin.math.hypot(src.x - back.x, src.y - back.y)
                    if (!dx.isFinite() || !dy.isFinite() || fbError > 1.8 ||
                        abs(dx) > 40 || abs(dy) > 40 ||
                        dst.x < 0 || dst.x >= next.cols() || dst.y < 0 || dst.y >= next.rows()) continue
                    validFrom.add(src)
                    validTo.add(dst)
                    shiftsX.add(dx)
                    shiftsY.add(dy)
                }

                if (validFrom.size >= 4) {
                    val medX = shiftsX.sorted()[shiftsX.size / 2]
                    val medY = shiftsY.sorted()[shiftsY.size / 2]
                    val residual = shiftsX.indices.map { abs(shiftsX[it] - medX) + abs(shiftsY[it] - medY) }.sorted()
                    if (residual[residual.size / 2] <= 4.5) {
                        translate = medX.toFloat() to medY.toFloat()
                    }

                    // Scale estimation requires features spread over the object;
                    // tight clusters can produce enormous unstable scale jumps.
                    val spreadX = validFrom.maxOf { it.x } - validFrom.minOf { it.x }
                    val spreadY = validFrom.maxOf { it.y } - validFrom.minOf { it.y }
                    if (validFrom.size >= 6 && spreadX >= coreRect.width * 0.28 &&
                        spreadY >= coreRect.height * 0.28) {
                        fittedFrom.fromArray(*validFrom.toTypedArray())
                        fittedTo.fromArray(*validTo.toTypedArray())
                        transform = org.opencv.calib3d.Calib3d.estimateAffinePartial2D(
                            fittedFrom, fittedTo, affineInliers,
                            org.opencv.calib3d.Calib3d.RANSAC, 2.5, 350L, 0.98, 5L
                        )
                        val a = transform
                        if (a != null && !a.empty() && a.rows() == 2 && a.cols() == 3) {
                            val mask = ByteArray(affineInliers.total().toInt())
                            affineInliers.get(0, 0, mask)
                            val nGood = mask.count { it.toInt() != 0 }
                            val candidateScale = kotlin.math.hypot(a.get(0, 0)[0], a.get(1, 0)[0])
                            if (nGood >= 5 && nGood * 2 >= validFrom.size &&
                                candidateScale in 0.88..1.14) {
                                val centerX = (t.left + t.right) * 0.5
                                val centerY = (t.top + t.bottom) * 0.5
                                val nextCX = a.get(0, 0)[0] * centerX + a.get(0, 1)[0] * centerY + a.get(0, 2)[0]
                                val nextCY = a.get(1, 0)[0] * centerX + a.get(1, 1)[0] * centerY + a.get(1, 2)[0]
                                val dx = (nextCX - centerX).toFloat()
                                val dy = (nextCY - centerY).toFloat()
                                // Reject inconsistent estimates rather than jump boxes.
                                if (translate == null ||
                                    (abs(dx - translate.first) < 6f && abs(dy - translate.second) < 6f)) {
                                    translate = dx to dy
                                    scale = candidateScale.toFloat()
                                }
                            }
                        }
                    }
                }
            }

            // A small, bounded template search helps if Lucas-Kanade finds only
            // stationary asphalt pixels inside a partly occluded vehicle box.
            val vx = translate?.first ?: 0f
            val vy = translate?.second ?: 0f
            val suspiciouslyStatic = abs(vx) + abs(vy) < 0.9f && appearanceChange > 8.0
            val matched = if (translate == null || suspiciouslyStatic)
                matchInterior(prev, next, coreRect) else null
            val motion = when {
                matched != null -> { scale = 1f; matched }
                translate != null && !suspiciouslyStatic -> translate
                else -> null
            } ?: return false

            if (abs(motion.first) + abs(motion.second) < 0.9f && appearanceChange > 8.0) return false
            val oldW = t.right - t.left
            val oldH = t.bottom - t.top
            val cx = (t.left + t.right) * 0.5f + motion.first
            val cy = (t.top + t.bottom) * 0.5f + motion.second
            val newW = oldW * scale
            val newH = oldH * scale
            if (!cx.isFinite() || !cy.isFinite() || !newW.isFinite() || !newH.isFinite()) return false
            if (newW < 7f || newH < 7f) return false
            val nx1 = cx - newW * 0.5f
            val ny1 = cy - newH * 0.5f
            val nx2 = cx + newW * 0.5f
            val ny2 = cy + newH * 0.5f
            if (nx2 <= 0f || ny2 <= 0f || nx1 >= next.cols() || ny1 >= next.rows()) return false
            t.left = nx1.coerceIn(0f, next.cols().toFloat())
            t.top = ny1.coerceIn(0f, next.rows().toFloat())
            t.right = nx2.coerceIn(0f, next.cols().toFloat())
            t.bottom = ny2.coerceIn(0f, next.rows().toFloat())
            return t.right > t.left && t.bottom > t.top
        } catch (_: Exception) {
            return false
        } finally {
            regionPrev.release(); regionNext.release(); diff.release()
            corners.release(); p0.release(); p1.release(); pBack.release()
            statusF.release(); statusB.release(); errorF.release(); errorB.release()
            affineInliers.release(); fittedFrom.release(); fittedTo.release()
            transform?.release()
        }
    }

    /** Local normalized-correlation fallback, limited to a small search window. */
    private fun matchInterior(prev: Mat, next: Mat, coreRect: Rect): Pair<Float, Float>? {
        val radius = 16
        val startX = (coreRect.x - radius).coerceAtLeast(0)
        val startY = (coreRect.y - radius).coerceAtLeast(0)
        val endX = (coreRect.x + coreRect.width + radius).coerceAtMost(next.cols())
        val endY = (coreRect.y + coreRect.height + radius).coerceAtMost(next.rows())
        val searchW = endX - startX
        val searchH = endY - startY
        if (coreRect.width < 5 || coreRect.height < 5 ||
            searchW < coreRect.width || searchH < coreRect.height) return null
        val reference = prev.submat(coreRect)
        val search = next.submat(Rect(startX, startY, searchW, searchH))
        val response = Mat()
        val mean = MatOfDouble()
        val std = MatOfDouble()
        return try {
            Core.meanStdDev(reference, mean, std)
            val contrast = std.toArray().firstOrNull() ?: return null
            if (contrast < 6.0) return null
            Imgproc.matchTemplate(search, reference, response, Imgproc.TM_CCOEFF_NORMED)
            val found = Core.minMaxLoc(response)
            if (!found.maxVal.isFinite() || found.maxVal < 0.78) return null
            val dx = (startX + found.maxLoc.x - coreRect.x).toFloat()
            val dy = (startY + found.maxLoc.y - coreRect.y).toFloat()
            if (abs(dx) > radius || abs(dy) > radius) null else dx to dy
        } catch (_: Exception) {
            null
        } finally {
            reference.release(); search.release(); response.release(); mean.release(); std.release()
        }
    }
}
