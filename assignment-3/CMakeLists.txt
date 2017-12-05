cmake_minimum_required(VERSION 3.8)

project(dnsmat)
    enable_language(C)
    find_package(MPI REQUIRED)
    include_directories(${MPI_INCLUDE_PATH})
    find_package(BLAS)
    find_package(LAPACK)
    if(LAPACK_FOUND AND BLAS_FOUND)
        set(lapackblas_libraries ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
    endif()

    set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -O3 -Wall -std=gnu99 -g3 -ggdb -lm")
    set(SOURCE_FILES main.backup matrix.c comm.c)
    add_executable(dnsmat ${SOURCE_FILES})
    target_link_libraries(dnsmat m
                        ${MPI_LIBRARIES}
                        ${lapackblas_libraries})




project(dgemm_test)
    enable_language(C)
    #set(CMAKE_C_COMPILER gcc)
    find_package(BLAS)
    find_package(LAPACK)
    if(LAPACK_FOUND AND BLAS_FOUND)
        set(lapackblas_libraries ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
    endif()

    set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -O3 -Wall -std=gnu99 -g3 -ggdb ")

    set(SOURCE_FILES dgemm_test.c)
    add_executable(dgemm_test ${SOURCE_FILES})
    target_link_libraries(dgemm_test ${lapackblas_libraries})
