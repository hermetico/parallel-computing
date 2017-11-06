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

cmake_minimum_required(VERSION 3.8)
project(mpi)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

SET(CMAKE_C_COMPILER mpic++)
SET(CMAKE_CXX_COMPILER mpicxx)

set(SOURCE_FILES mpi.cpp common.cpp)
add_executable(mpi ${SOURCE_FILES})