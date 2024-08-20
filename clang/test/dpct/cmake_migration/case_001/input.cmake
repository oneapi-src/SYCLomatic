CMake_minimum_required(VERSION 3.10)
###CMake_minimum_required(VERSION 3.10)
#CMake_minimum_required(VERSION 3.10)
Project(foo-bar LANGUAGES CXX CUDA)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
Find_Package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
Find_Package(CUDA)
set(SOURCES
    ${CMAKE_SOURCE_DIR}/foo/main.cu
    ${CMAKE_SOURCE_DIR}/foo/bar/util.cu
)
include_directorieS(
    ${CMAKE_SOURCE_DIR}/foo/bar
    ${CUDA_INCLUDE_DIRS}
)
add_executable(foo-bar ${SOURCES})
project(foo CUDA)

project(foo2 CUDA CXX)

set(SOURCES a.cu b.cuh)

project  (foo2 CUDA CXX)

project  (  foo2 CUDA CXX)

if(FOO_OPENMP)
    if(NOT OPENMP_FOUND)
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            message(FATAL_ERROR "The compiler you are using does not support OpenMP parallelism, "
                "You can disable OpenMP in Gromacs with -DFOO_OPENMP=OFF, but instead "
                "we recommend installing the unsupported library distributed by the R "
                "project from https://mac.r-project.org/openmp/ - or switch to gcc.")
        else()
            message(FATAL_ERROR "The compiler you are using does not support OpenMP parallelism. "
                "This might hurt your performance a lot, in particular with GPUs. "
                "Try using a more recent version, or a different compiler. "
                "If you don't want to use OpenMP, disable it explicitly with -DFOO_OPENMP=OFF")
        endif()
    endif()
endif()

if(CUDA_NVCC_EXECUTABLE)
endif()
