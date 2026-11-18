cmake_minimum_required(VERSION 3.8)
project(unina_perception_dla LANGUAGES CXX CUDA)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA Configuration for Jetson Orin (sm_87)
# Note: For non-Orin GPUs (workstation/Colab/Kaggle), override with: -DCMAKE_CUDA_ARCHITECTURES=native
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 87)
endif()

# ============================================================================
# Core Dependencies (Standard C++ / CUDA / TensorRT / OpenCV)
# ============================================================================
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

# TensorRT
find_library(NVINFER_LIB nvinfer)
find_library(NVPARSERS_LIB nvparsers)
if(NOT NVINFER_LIB)
    message(FATAL_ERROR "nvinfer library not found. Please set CMAKE_LIBRARY_PATH.")
endif()

include_directories(
    ${CUDAToolkit_INCLUDE_DIRS}
    include
)
if(EXISTS "/usr/include/aarch64-linux-gnu")
    include_directories(/usr/include/aarch64-linux-gnu)
endif()

# ============================================================================
# Shared Sources
# ============================================================================
set(CUDA_SOURCES
    src/CudaPreprocess.cu
    src/CudaPostprocess.cu
)
set(CORE_SOURCES
    src/UninaPerception.cpp
    src/FileImageSource.cpp
)

set(TRT_LIBS ${NVINFER_LIB})
if(NVPARSERS_LIB)
    list(APPEND TRT_LIBS ${NVPARSERS_LIB})
endif()

# ============================================================================
# Target 1: unina_perception (Standalone single-image inference)
# ============================================================================
add_executable(unina_perception
    main.cpp
    ${CORE_SOURCES}
    ${CUDA_SOURCES}
)
target_link_libraries(unina_perception
    ${OpenCV_LIBS}
    CUDA::cudart
    ${TRT_LIBS}
)
add_executable(unina_perception_standalone ALIAS unina_perception)

# ============================================================================
# Target 2: unina_perception_offline (Batch directory / video testing)
# ============================================================================
add_executable(unina_perception_offline
    main_offline.cpp
    ${CORE_SOURCES}
    ${CUDA_SOURCES}
)
target_link_libraries(unina_perception_offline
    ${OpenCV_LIBS}
    CUDA::cudart
    ${TRT_LIBS}
)

# ============================================================================
# Installation Rules
# ============================================================================
install(TARGETS unina_perception unina_perception_offline
    RUNTIME DESTINATION bin
)
install(FILES config.yaml
    DESTINATION share/${PROJECT_NAME}
)
