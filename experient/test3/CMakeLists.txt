cmake_minimum_required(VERSION 3.10)
project(NoiseAnalysis)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    src
)

# Source files
set(SOURCES
    src/main.cpp
    src/noise_base.cpp
    src/perlin_noise.cpp
    src/sparse_convolution_noise.cpp
    src/wavelet_noise.cpp
    src/anisotropic_noise.cpp
    src/better_gradient_noise.cpp
    src/gabor_noise.cpp
    src/noise_analyzer.cpp
)

# Create executable
add_executable(noise_analysis ${SOURCES})

# Link libraries
target_link_libraries(noise_analysis
    ${OpenCV_LIBS}
    Eigen3::Eigen
)