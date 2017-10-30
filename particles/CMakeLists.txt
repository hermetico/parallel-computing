cmake_minimum_required(VERSION 3.8)
# Serial implementation
project(Serial)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp -fPIC -O0 -g3 -ggdb")
set(SOURCE_FILES serial.cpp common.cpp)
add_executable(Serial ${SOURCE_FILES})

cmake_minimum_required(VERSION 3.8)
# OpenMP implementation
project(OpenMP)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp -fPIC -O0 -g3 -ggdb")
set(SOURCE_FILES openmp.cpp common.cpp)
add_executable(OpenMP ${SOURCE_FILES})
