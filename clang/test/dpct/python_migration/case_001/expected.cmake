cmake_minimum_required(VERSION 3.24)
project(foo-bar LANGUAGES CXX )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
include(dpct.cmake)

find_package(IntelSYCL REQUIRED)
find_package(IntelSYCL REQUIRED)
