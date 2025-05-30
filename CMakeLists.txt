cmake_minimum_required(VERSION 3.10)


project(hw1-ppm)

set(CMAKE_CXX_FLAGS "-g -Wall")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_executable(hw1-ppm main.cpp)