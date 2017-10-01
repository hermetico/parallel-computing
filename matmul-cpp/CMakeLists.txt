cmake_minimum_required(VERSION 3.8)
# CustomDGEMM configuration
project(CustomDGEMM)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp -fPIC -O0 -g3 -ggdb")

find_package(BLAS)
find_package(LAPACK)
if(LAPACK_FOUND AND BLAS_FOUND)
    set(lapackblas_libraries ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()

set(SOURCE_FILES dgemm.cpp benchmark.cpp)
add_executable(CustomDGEMM ${SOURCE_FILES})
target_link_libraries(CustomDGEMM ${lapackblas_libraries})


# DetailedNaive configuration
project(DetailedNaive)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp -fPIC -O0 -g3 -ggdb")

find_package(BLAS)
find_package(LAPACK)
if(LAPACK_FOUND AND BLAS_FOUND)
    set(lapackblas_libraries ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()


set(SOURCE_FILES dgemm-naive.cpp detailed_benchmark.cpp)
add_executable(DetailedNaive ${SOURCE_FILES})
target_link_libraries(DetailedNaive ${lapackblas_libraries})


# DetailedCustomDGEMM configuration
project(DetailedCustomDGEMM)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp -fPIC -O0 -g3 -ggdb")

find_package(BLAS)
find_package(LAPACK)
if(LAPACK_FOUND AND BLAS_FOUND)
    set(lapackblas_libraries ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()


set(SOURCE_FILES dgemm.cpp detailed_benchmark.cpp)
add_executable(DetailedCustomDGEMM ${SOURCE_FILES})
target_link_libraries(DetailedCustomDGEMM ${lapackblas_libraries})

# BlockedDGEMM configuration
project(BlockedDGEMM)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

find_package(BLAS)
find_package(LAPACK)
if(LAPACK_FOUND AND BLAS_FOUND)
    set(lapackblas_libraries ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()

set(SOURCE_FILES dgemm-blocked.cpp benchmark.cpp)
add_executable(BlockedDGEMM ${SOURCE_FILES})
target_link_libraries(BlockedDGEMM ${lapackblas_libraries})


# DetailedBlockedDGEMM configuration
project(DetailedBlockedDGEMM)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

find_package(BLAS)
find_package(LAPACK)
if(LAPACK_FOUND AND BLAS_FOUND)
    set(lapackblas_libraries ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()

set(SOURCE_FILES dgemm-blocked.cpp detailed_benchmark.cpp)
add_executable(DetailedBlockedDGEMM ${SOURCE_FILES})
target_link_libraries(DetailedBlockedDGEMM ${lapackblas_libraries})

# TESTS configuration
project(TESTS C)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp -fPIC -O0 -g3 -ggdb")

set(SOURCE_FILES dgemm.cpp tests.cpp)
add_executable(TESTS ${SOURCE_FILES})