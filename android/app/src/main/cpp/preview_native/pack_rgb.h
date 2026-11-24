#pragma once
#include <cstdint>
#include <cstddef>
#if !defined(PREVIEW_FORCE_SCALAR) && defined(__ARM_NEON)
#include <arm_neon.h>
#define PREVIEW_NEON 1
#elif !defined(PREVIEW_FORCE_SCALAR) && defined(__SSE2__)
#include <emmintrin.h>
#define PREVIEW_SSE2 1
#endif

namespace preview {
// The MobileIE graph clips all finite output values to [0,1]. Also handle NaN/Inf
// defensively; NaN maps to 0. Use truncation + fractional comparison rather than
// adding 0.5f before truncation (which can round a near-half float upward).
inline uint32_t rounded(float x) {
    if (!(x > 0.f)) return 0;
    if (x >= 1.f) return 255;
    const float p = x * 255.f;
    const auto n = static_cast<uint32_t>(p);
    return n + static_cast<uint32_t>(p - static_cast<float>(n) >= .5f);
}
#if PREVIEW_NEON
inline uint32x4_t quantize(float32x4_t x) {
    const auto zero = vdupq_n_f32(0.f);
    x = vbslq_f32(vcgtq_f32(x, zero), x, zero);
    x = vminq_f32(x, vdupq_n_f32(1.f));
    auto p = vmulq_n_f32(x, 255.f);
    auto n = vcvtq_u32_f32(p);
    auto frac = vsubq_f32(p, vcvtq_f32_u32(n));
    return vaddq_u32(n, vandq_u32(vcgeq_f32(frac, vdupq_n_f32(.5f)), vdupq_n_u32(1)));
}
#elif PREVIEW_SSE2
inline __m128i quantize(__m128 x) {
    x = _mm_min_ps(_mm_max_ps(x, _mm_setzero_ps()), _mm_set1_ps(1.f));
    auto p = _mm_mul_ps(x, _mm_set1_ps(255.f));
    auto n = _mm_cvttps_epi32(p);
    auto frac = _mm_sub_ps(p, _mm_cvtepi32_ps(n));
    return _mm_add_epi32(n, _mm_and_si128(_mm_castps_si128(_mm_cmpge_ps(frac, _mm_set1_ps(.5f))), _mm_set1_epi32(1)));
}
#endif

inline void pack(const float* rgb, int stride, int left, int top,
                 int width, int height, uint32_t* dst) {
    const size_t plane = static_cast<size_t>(stride) * stride;
    for (int y = 0; y < height; ++y) {
        const size_t off = static_cast<size_t>(top + y) * stride + left;
        const float* r = rgb + off;
        const float* g = r + plane;
        const float* b = g + plane;
        int x = 0;
#if PREVIEW_NEON
        for (; x + 4 <= width; x += 4) {
            auto rr = vshlq_n_u32(quantize(vld1q_f32(r + x)), 16);
            auto gg = vshlq_n_u32(quantize(vld1q_f32(g + x)), 8);
            auto bb = quantize(vld1q_f32(b + x));
            vst1q_u32(dst + x, vorrq_u32(vdupq_n_u32(0xff000000u), vorrq_u32(rr, vorrq_u32(gg, bb))));
        }
#elif PREVIEW_SSE2
        for (; x + 4 <= width; x += 4) {
            auto rr = _mm_slli_epi32(quantize(_mm_loadu_ps(r + x)), 16);
            auto gg = _mm_slli_epi32(quantize(_mm_loadu_ps(g + x)), 8);
            auto bb = quantize(_mm_loadu_ps(b + x));
            auto pixel = _mm_or_si128(_mm_set1_epi32(static_cast<int32_t>(0xff000000u)), _mm_or_si128(rr, _mm_or_si128(gg, bb)));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + x), pixel);
        }
#endif
        for (; x < width; ++x) {
            dst[x] = 0xff000000u | (rounded(r[x]) << 16) | (rounded(g[x]) << 8) | rounded(b[x]);
        }
        dst += width;
    }
}
}
