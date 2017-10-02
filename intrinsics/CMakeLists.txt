cmake_minimum_required(VERSION 3.8)
project(intrinsics)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp -fPIC -O0 -g3 -ggdb -mavx")
set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES hello_avx.cpp)
add_executable(intrinsics ${SOURCE_FILES})


cmake_minimum_required(VERSION 3.8)
project(load_dobules)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp -fPIC -O0 -g3 -ggdb -mavx")
set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES hello_load.cpp)
add_executable(load_dobules ${SOURCE_FILES})


cmake_minimum_required(VERSION 3.8)
project(simple_matmul)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp -fPIC -O0 -g3 -ggdb -mavx")
set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES simple-matmul.cpp)
add_executable(simple_matmul ${SOURCE_FILES})

cmake_minimum_required(VERSION 3.8)
project(fma_matmul)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp -fPIC -O0 -g3 -ggdb -mavx2 -mfma")
set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES fma-matmul.cpp)
add_executable(fma_matmul ${SOURCE_FILES})