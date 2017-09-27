# CustomDGEMM configuration
project(CustomDGEMM C)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O1 -Wall -std=gnu99")

find_package(BLAS)
find_package(LAPACK)
if(LAPACK_FOUND AND BLAS_FOUND)
    set(lapackblas_libraries ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()

set(SOURCE_FILES dgemm.c benchmark.c)
add_executable(CustomDGEMM ${SOURCE_FILES})
target_link_libraries(CustomDGEMM ${lapackblas_libraries})

# BlockedDGEMM configuration
project(BlockedDGEMM C)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O1 -Wall -std=gnu99")

find_package(BLAS)
find_package(LAPACK)
if(LAPACK_FOUND AND BLAS_FOUND)
    set(lapackblas_libraries ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()

set(SOURCE_FILES dgemm-blocked.c benchmark.c)
add_executable(BlockedDGEMM ${SOURCE_FILES})
target_link_libraries(BlockedDGEMM ${lapackblas_libraries})


# TESTS configuration
project(TESTS C)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu99")

set(SOURCE_FILES dgemm.c tests.c)
add_executable(TESTS ${SOURCE_FILES})