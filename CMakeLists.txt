cmake_minimum_required(VERSION 3.16)
project(yolov5-tensorflow)



find_package(OpenCV REQUIRED)

# We installed Tensorflow in /opt/tensorflow - change if you installed elsewhere
set(TENSORFLOW_LIB_DIR "/opt/tensorflow/lib")

file(GLOB SOURCES
    yolov5.h
    yolov5.cpp
    #main.cpp
    video.cpp
)

add_executable(${CMAKE_PROJECT_NAME} ${SOURCES})#get_prediction

target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${TENSORFLOW_LIB_DIR}/include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${CMAKE_PROJECT_NAME} ${TENSORFLOW_LIB_DIR}/libtensorflow_cc.so ${TENSORFLOW_LIB_DIR}/libtensorflow_framework.so ${OpenCV_LIBRARIES})