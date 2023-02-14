cmake_minimum_required(VERSION 3.0)
project(TDTLidar_RM2023)

# set(CMAKE_CXX_STANDARD 17)
# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -w ")
set(PROJECT_ROOT_PATH "${PROJECT_SOURCE_DIR}")
set(EXECUTABLE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/bin/")
set(LIBRARY_OUTPUT_PATH "${PROJECT_BINARY_DIR}/lib/")
link_directories(${PROJECT_BINARY_DIR}/lib/)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# cuda
find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# opencv
find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# 开启Opencv的TBB加速
add_definitions(-D HAVE_TBB)
# 开启Opencv的IPP加速
add_definitions(-D HAVE_IPP)

# PCL
# set(PCL_DIR "/usr/local/pcl-1.9/share/pcl-1.9")
find_package(PCL  REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
# Boost
find_package(Boost 1.54 REQUIRED COMPONENTS system)
# yaml_cpp
find_package(yaml-cpp REQUIRED)
include_directories(${yaml-cpp_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/yolov5)
add_subdirectory(yolov5)
include(FindPkgConfig) # 加载FindPkgConfig功能
pkg_check_modules(XCB xcb REQUIRED) # 查找xcb库
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
  message("OPENMP FOUND")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS
      "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
link_directories(/usr/lib/x86_64-linux-gnu/lib/)
# 找到qt
find_package(
  Qt5
  COMPONENTS Widgets
  REQUIRED)

add_executable(
  TDTLidar_RM2023
  main.cpp
  ${LOCATE_SRCS}
  ${USART_SRCS}
)

target_link_libraries(
  TDTLidar_RM2023
  PUBLIC ${OpenCV_LIBS}
         myplugins
         cudart
         ${PCL_LIBRARIES}
         apr-1
         -lpthread
         -lm
         -mavx2
         Boost::boost
         Boost::system
         # -lpython3.9 用来找python3.9-config --includes --libs
         -lcrypt
         -lpthread
         -ldl
         -lutil
         -lm
         -lm
         # libpython3.6.so
         ${XCB_LDFLAGS})

target_include_directories(TDTLidar_RM2023 PUBLIC /usr/include/apr-1.0)


# add_executable(coor ${LOG_SRCS} 相机雷达联合标定.cpp ${CONFIG_SRCS})
# find_package(Ceres REQUIRED)

# target_link_libraries(coor ${OpenCV_LIBS} tdtcamera tdtcommon ${CERES_LIBRARIES})
