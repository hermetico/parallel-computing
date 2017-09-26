project(dgemm C)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O1 -Wall -std=gnu99")

set(SOURCE_FILES dgemm.c benchmark.c )
add_executable(dgemm-benchmark ${SOURCE_FILES})