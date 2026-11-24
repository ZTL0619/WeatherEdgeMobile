#include <jni.h>
#include <cstdint>
#include "pack_rgb.h"

extern "C" JNIEXPORT void JNICALL
Java_com_example_weatheredgemobile_NativePreview_packRgb(
    JNIEnv* env, jclass, jfloatArray rgb, jint stride, jint left, jint top,
    jint width, jint height, jintArray argb) {
    const int64_t plane = static_cast<int64_t>(stride) * stride;
    if (!rgb || !argb || stride <= 0 || left < 0 || top < 0 || width <= 0 || height <= 0 ||
        static_cast<int64_t>(left) + width > stride || static_cast<int64_t>(top) + height > stride ||
        plane > env->GetArrayLength(rgb) / 3 ||
        static_cast<int64_t>(width) * height > env->GetArrayLength(argb)) {
        auto cls = env->FindClass("java/lang/IllegalArgumentException");
        if (cls) env->ThrowNew(cls, "Invalid CHW RGB buffer or preview ROI");
        return;
    }
    auto* src = env->GetFloatArrayElements(rgb, nullptr);
    if (!src) return; // JVM has reported allocation failure.
    auto* dst = env->GetIntArrayElements(argb, nullptr);
    if (!dst) {
        env->ReleaseFloatArrayElements(rgb, src, JNI_ABORT);
        return;
    }
    static_assert(sizeof(jint) == sizeof(uint32_t), "ARGB must be 32-bit");
    preview::pack(src, stride, left, top, width, height, reinterpret_cast<uint32_t*>(dst));
    env->ReleaseIntArrayElements(argb, dst, 0);
    env->ReleaseFloatArrayElements(rgb, src, JNI_ABORT);
}

#include <new>
#include "resize_rgb.h"

extern "C" JNIEXPORT void JNICALL
Java_com_example_weatheredgemobile_NativePreview_resizeLetterbox(
    JNIEnv* env, jclass, jfloatArray rgb, jint stride, jint left, jint top,
    jint width, jint height, jfloatArray output, jint size, jint dstLeft,
    jint dstTop, jint dstWidth, jint dstHeight) {
    const int64_t srcPlane = static_cast<int64_t>(stride) * stride;
    const int64_t dstPlane = static_cast<int64_t>(size) * size;
    if (!rgb || !output || env->IsSameObject(rgb, output) || stride <= 0 || size <= 0 ||
        left < 0 || top < 0 || width <= 0 || height <= 0 ||
        dstLeft < 0 || dstTop < 0 || dstWidth <= 0 || dstHeight <= 0 ||
        static_cast<int64_t>(left) + width > stride || static_cast<int64_t>(top) + height > stride ||
        static_cast<int64_t>(dstLeft) + dstWidth > size || static_cast<int64_t>(dstTop) + dstHeight > size ||
        srcPlane > env->GetArrayLength(rgb) / 3 || dstPlane > env->GetArrayLength(output) / 3) {
        if (auto cls = env->FindClass("java/lang/IllegalArgumentException"))
            env->ThrowNew(cls, "Invalid float32 letterbox buffers or ROI");
        return;
    }
    auto* src = env->GetFloatArrayElements(rgb, nullptr);
    if (!src) return;
    auto* dst = env->GetFloatArrayElements(output, nullptr);
    if (!dst) {
        env->ReleaseFloatArrayElements(rgb, src, JNI_ABORT);
        return;
    }
    bool ok = false;
    try {
        preview::resizeLetterbox(src, stride, left, top, width, height, dst, size,
                                 dstLeft, dstTop, dstWidth, dstHeight);
        ok = true;
    } catch (const std::bad_alloc&) {
        // Release JNI arrays before reporting memory failure to Java.
    }
    env->ReleaseFloatArrayElements(output, dst, ok ? 0 : JNI_ABORT);
    env->ReleaseFloatArrayElements(rgb, src, JNI_ABORT);
    if (!ok) {
        if (auto cls = env->FindClass("java/lang/OutOfMemoryError"))
            env->ThrowNew(cls, "Native letterbox scratch allocation failed");
    }
}
