cmake_minimum_required(VERSION 3.8)
project(intrinsics)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp -fPIC -O0 -g3 -ggdb -mavx")
set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES hello_avx.cpp)
add_executable(intrinsics ${SOURCE_FILES})