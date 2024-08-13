set(CUDA_HOST_COMPILER "clang")
set (CUDA_HOST_COMPILER "clang")

set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE FILEPATH "Resetting the value for host compiler")

set(CUDA_HOST_FLAGS "--extra-warnings -Wdeprecated")
set (CUDA_HOST_FLAGS "--extra-warnings -Wdeprecated")

set(CUDA_HOST_FLAGS "${host_flags}")

list(APPEND CMAKE_CUDA_FLAGS "-Xcompiler=-Wno-float-conversion")
list(APPEND CMAKE_CUDA_FLAGS "-Xcompiler=-fno-strict-aliasing")
list(APPEND CMAKE_CUDA_FLAGS "-Xcudafe=--diag_suppress=unrecognized_gcc_pragma")
list(APPEND CMAKE_CUDA_FLAGS "--extended-lambda")
list(APPEND CMAKE_CUDA_FLAGS "--expt-relaxed-constexpr")

if (NOT CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
  message(FATAL_ERROR "Error msg.")
endif()
