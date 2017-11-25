cmake_minimum_required(VERSION 3.8)
project(dnsmat)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

set(CMAKE_C_COMPILER mpic++)
set(CMAKE_CXX_COMPILER mpicxx)
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -O3 -Wall -std=gnu99 -g3 -ggdb")

set(SOURCE_FILES main.c matrix.c comm.c)
add_executable(dnsmat ${SOURCE_FILES})