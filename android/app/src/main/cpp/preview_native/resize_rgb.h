#pragma once
#include <algorithm>
#include <cstddef>
#include <vector>
#include "pack_rgb.h"  // Shared SSE2 / NEON selection.

namespace preview {
struct AxisPoint {
    int lo, hi;
    float weight;
};

inline std::vector<AxisPoint> resizeAxis(int src, int dst) {
    std::vector<AxisPoint> points(static_cast<size_t>(dst));
    for (int i = 0; i < dst; ++i) {
        // Same float32 half-pixel formula as the original Kotlin FloatLetterbox.
        const float f = std::clamp((static_cast<float>(i) + .5f) * src / dst - .5f,
                                   0.f, static_cast<float>(src - 1));
        const int a = static_cast<int>(f); // f >= 0, equivalent to floor.
        points[i] = {a, std::min(a + 1, src - 1), f - static_cast<float>(a)};
    }
    return points;
}

// Resize horizontal rows once, and vectorize the vertical interpolation.
// Do not fuse multiply-add: CMake already uses -ffp-contract=off.
inline void horizontal(const float* row, float* dst, const AxisPoint* axis, int width) {
    for (int x = 0; x < width; ++x) {
        const auto& p = axis[x];
        const float a = row[p.lo];
        dst[x] = a + (row[p.hi] - a) * p.weight;
    }
}

inline void vertical(const float* a, const float* b, float fy, float* dst, int width) {
    int x = 0;
#if PREVIEW_NEON
    const auto f = vdupq_n_f32(fy);
    for (; x + 4 <= width; x += 4) {
        const auto av = vld1q_f32(a + x);
        const auto bv = vld1q_f32(b + x);
        vst1q_f32(dst + x, vaddq_f32(av, vmulq_f32(vsubq_f32(bv, av), f)));
    }
#elif PREVIEW_SSE2
    const auto f = _mm_set1_ps(fy);
    for (; x + 4 <= width; x += 4) {
        const auto av = _mm_loadu_ps(a + x);
        const auto bv = _mm_loadu_ps(b + x);
        _mm_storeu_ps(dst + x, _mm_add_ps(av, _mm_mul_ps(_mm_sub_ps(bv, av), f)));
    }
#endif
    for (; x < width; ++x) dst[x] = a[x] + (b[x] - a[x]) * fy;
}

inline void resizeLetterbox(const float* src, int stride, int left, int top,
                            int width, int height, float* dst, int size,
                            int dstLeft, int dstTop, int dstWidth, int dstHeight) {
    const auto xs = resizeAxis(width, dstWidth);
    const auto ys = resizeAxis(height, dstHeight);
    // Reuse two horizontal rows across vertical samples. The lookup is monotonic.
    std::vector<float> scratch(static_cast<size_t>(2) * dstWidth);
    const size_t srcPlane = static_cast<size_t>(stride) * stride;
    const size_t dstPlane = static_cast<size_t>(size) * size;
    std::fill_n(dst, 3 * dstPlane, 114.f / 255.f);
    for (int c = 0; c < 3; ++c) {
        const float* base = src + c * srcPlane + static_cast<size_t>(top) * stride + left;
        float* out = dst + c * dstPlane + static_cast<size_t>(dstTop) * size + dstLeft;
        float* rowA = scratch.data();
        float* rowB = rowA + dstWidth;
        int idA = -1, idB = -1;
        for (int y = 0; y < dstHeight; ++y) {
            const auto& p = ys[y];
            if (idA != p.lo) {
                if (idB == p.lo) {
                    std::swap(rowA, rowB);
                    std::swap(idA, idB);
                } else {
                    horizontal(base + static_cast<size_t>(p.lo) * stride, rowA, xs.data(), dstWidth);
                    idA = p.lo;
                }
            }
            const float* second = rowA;
            if (p.hi != p.lo) {
                if (idB != p.hi) {
                    horizontal(base + static_cast<size_t>(p.hi) * stride, rowB, xs.data(), dstWidth);
                    idB = p.hi;
                }
                second = rowB;
            }
            vertical(rowA, second, p.weight, out + static_cast<size_t>(y) * size, dstWidth);
        }
    }
}
} // namespace preview
