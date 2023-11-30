CMake_minimum_required(VERSION 3.10)
###CMake_minimum_required(VERSION 3.10)
#CMake_minimum_required(VERSION 3.10)
Project(foo-bar LANGUAGES CXX CUDA)
set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
Find_Package(CUDA REQUIRED)
set(SOURCES
    ${CMAKE_SOURCE_DIR}/foo/main.cu
    ${CMAKE_SOURCE_DIR}/foo/bar/util.cu
)
include_directorieS(
    ${CMAKE_SOURCE_DIR}/foo/bar
    ${CUDA_INCLUDE_DIRS}
)
add_executable(foo-bar ${SOURCES})

