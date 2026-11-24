cmake_minimum_required(VERSION 3.18)
project(weatheredge_preview LANGUAGES CXX)
add_library(weatheredge_preview SHARED preview_jni.cpp)
target_compile_features(weatheredge_preview PRIVATE cxx_std_17)
# Keep the same multiply-then-round behavior as Kotlin; no fast-math.
target_compile_options(weatheredge_preview PRIVATE -O3 -ffp-contract=off -fno-fast-math)
