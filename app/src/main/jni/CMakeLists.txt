project(yolov5ncnn)

cmake_minimum_required(VERSION 3.10)


set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-4.5.1-android-sdk/sdk/native/jni)
find_package(OpenCV REQUIRED core imgproc)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20220420-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_library(yolosegment SHARED yolosegmentncnn_jni.cpp)

target_link_libraries(yolosegment
        ncnn
        jnigraphics
        ${OpenCV_LIBS}
        )
