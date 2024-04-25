set(CUDA_HOST_COMPILER "clang")
set (CUDA_HOST_COMPILER "clang")

set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE FILEPATH "Resetting the value for host compiler")

set(CUDA_HOST_FLAGS "--extra-warnings -Wdeprecated")
set (CUDA_HOST_FLAGS "--extra-warnings -Wdeprecated")

set(CUDA_HOST_FLAGS "${host_flags}")
