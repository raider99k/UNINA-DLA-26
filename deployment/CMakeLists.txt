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
# Note: For VM builds, override with: -DCMAKE_CUDA_ARCHITECTURES=native
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 87)
endif()

# ============================================================================
# Build Options (Strictly Optional)
# ============================================================================
option(USE_ROS "Enable ROS 2 Humble integration" OFF)
option(USE_ZED "Enable ZED SDK integration" OFF)

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

# ============================================================================
# Optional Integration: ROS 2 Humble
# ============================================================================
if(USE_ROS)
    find_package(ament_cmake REQUIRED)
    find_package(rclcpp REQUIRED)
    find_package(std_msgs REQUIRED)
    find_package(geometry_msgs REQUIRED)
    find_package(visualization_msgs REQUIRED)
    add_compile_definitions(USE_ROS)
endif()

# ============================================================================
# Optional Integration: ZED SDK
# ============================================================================
if(USE_ZED)
    find_package(ZED 4 REQUIRED)
    add_compile_definitions(USE_ZED_SDK)
endif()

include_directories(
    ${CUDAToolkit_INCLUDE_DIRS}
    include
)
if(EXISTS "/usr/include/aarch64-linux-gnu")
    include_directories(/usr/include/aarch64-linux-gnu)
endif()

# ============================================================================
# Shared Sources (Zero ROS / Zero ZED dependencies)
# ============================================================================
set(CUDA_SOURCES
    src/CudaPreprocess.cu
    src/CudaPostprocess.cu
)
set(CORE_SOURCES
    src/UninaPerception.cpp
    src/Perception3D.cpp
    src/FileImageSource.cpp
)

# ============================================================================
# Target 1: unina_perception (Standalone image-file mode)
# ============================================================================
add_executable(unina_perception 
    main.cpp 
    ${CORE_SOURCES}
    ${CUDA_SOURCES}
)
target_link_libraries(unina_perception
    ${OpenCV_LIBS}
    CUDA::cudart
    ${NVINFER_LIB}
    ${NVPARSERS_LIB}
)
add_executable(unina_perception_standalone ALIAS unina_perception)

# ============================================================================
# Target 2: unina_perception_offline (Standalone batch directory/video testing)
# ============================================================================
add_executable(unina_perception_offline
    main_offline.cpp
    ${CORE_SOURCES}
    ${CUDA_SOURCES}
)
target_link_libraries(unina_perception_offline
    ${OpenCV_LIBS}
    CUDA::cudart
    ${NVINFER_LIB}
    ${NVPARSERS_LIB}
)

# ============================================================================
# Target 3: unina_perception_zed (Live ZED X camera with 3D, no ROS)
# ============================================================================
if(USE_ZED)
    add_executable(unina_perception_zed 
        main_zed.cpp 
        ${CORE_SOURCES}
        src/ZedImageSource.cpp
        ${CUDA_SOURCES}
    )
    target_link_libraries(unina_perception_zed
        ${OpenCV_LIBS}
        CUDA::cudart
        ${NVINFER_LIB}
        ${NVPARSERS_LIB}
        ${ZED_LIBRARIES}
    )
    target_include_directories(unina_perception_zed PRIVATE ${ZED_INCLUDE_DIRS})
endif()

# ============================================================================
# Target 4: unina_perception_ros (ROS 2 Humble Node)
# ============================================================================
if(USE_ROS AND USE_ZED)
    add_executable(unina_perception_ros
        main_ros.cpp
        ${CORE_SOURCES}
        src/ZedImageSource.cpp
        ${CUDA_SOURCES}
    )
    
    ament_target_dependencies(unina_perception_ros
        rclcpp
        std_msgs
        geometry_msgs
        visualization_msgs
    )
    
    target_link_libraries(unina_perception_ros
        ${OpenCV_LIBS}
        CUDA::cudart
        ${NVINFER_LIB}
        ${NVPARSERS_LIB}
        ${ZED_LIBRARIES}
    )
    
    target_include_directories(unina_perception_ros PRIVATE ${ZED_INCLUDE_DIRS})
endif()

# ============================================================================
# Installation Rules
# ============================================================================
if(USE_ROS)
    install(TARGETS unina_perception unina_perception_offline
        DESTINATION lib/${PROJECT_NAME}
    )
    if(USE_ZED)
        install(TARGETS unina_perception_zed
            DESTINATION lib/${PROJECT_NAME}
        )
        if(TARGET unina_perception_ros)
            install(TARGETS unina_perception_ros
                DESTINATION lib/${PROJECT_NAME}
            )
        endif()
    endif()

    install(FILES config.yaml
        DESTINATION share/${PROJECT_NAME}
    )
    ament_package()
else()
    install(TARGETS unina_perception unina_perception_offline
        RUNTIME DESTINATION bin
    )
    if(USE_ZED)
        install(TARGETS unina_perception_zed
            RUNTIME DESTINATION bin
        )
    endif()

    install(FILES config.yaml
        DESTINATION share/${PROJECT_NAME}
    )
endif()

