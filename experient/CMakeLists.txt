
cmake_minimum_required(VERSION 3.10)
project(WaveletNoise)

# 設定 C++ 標準
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 主要執行檔
add_executable(wavelet_noise wavelet_noise.cpp)

# 如果需要數學函式庫（Linux/Unix）
if(UNIX)
    target_link_libraries(wavelet_noise m)
endif()

# 設定編譯器警告
if(MSVC)
    target_compile_options(wavelet_noise PRIVATE /W4)
else()
    target_compile_options(wavelet_noise PRIVATE -Wall -Wextra -Wpedantic)
endif()

# 優化選項（Release 模式）
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        target_compile_options(wavelet_noise PRIVATE /O2)
    else()
        target_compile_options(wavelet_noise PRIVATE -O3)
    endif()
endif()