cmake_minimum_required(VERSION 2.6)

project(yolov5)

add_definitions(-std=c++14)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 14)
set(CUDA_INCLUDE_DIRS "/usr/local/cuda/targets/x86_64-linux/include/")
set(CUDA_CUDART_LIBRARY "/usr/local/cuda/targets/x86_64-linux/lib/")

find_package(CUDA REQUIRED)

#头文件搜索引擎
#include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
include_directories(/usr/local/cuda/targets/x86_64-linux/include)
link_directories(/usr/lib/x86_64-linux-gnu/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/yololayer.cu preprocess.cu yolov5_predictor.cpp resnet.h resnet18.cpp)
target_link_libraries(myplugins nvinfer cudart)


#find_package(OpenCV)
#include_directories(OpenCV_INCLUDE_DIRS)

#add_executable(yolov5 ${PROJECT_SOURCE_DIR}/yolov5_predictor.cpp)
#target_link_libraries(yolov5 nvinfer)
#target_link_libraries(yolov5 cudart)
#target_link_libraries(yolov5 myplugins)
#target_link_libraries(yolov5 ${OpenCV_LIBS})

add_definitions(-O2 -pthread)

