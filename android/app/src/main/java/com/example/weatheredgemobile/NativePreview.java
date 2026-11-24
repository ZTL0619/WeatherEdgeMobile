package com.example.weatheredgemobile;

/** Native RGB packing and float32 ROI letterboxing. */
public final class NativePreview {
    private static boolean loaded;
    public static synchronized void ensureLoaded() {
        if (loaded) return;
        try {
            System.loadLibrary("weatheredge_preview");
            loaded = true;
        } catch (UnsatisfiedLinkError e) {
            throw new IllegalStateException("Cannot load libweatheredge_preview.so; check CMake/ABI configuration", e);
        }
    }
    private NativePreview() {}
    public static native void packRgb(float[] rgb, int stride, int left, int top,
                                      int width, int height, int[] argb);
    public static native void resizeLetterbox(float[] rgb, int stride, int left, int top,
                                              int width, int height, float[] output, int size,
                                              int dstLeft, int dstTop, int dstWidth, int dstHeight);
}
