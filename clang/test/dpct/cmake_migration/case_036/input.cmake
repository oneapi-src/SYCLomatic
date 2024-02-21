set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options \"-fvisibility=hidden -Wno-free-nonheap-object\" --Wno-deprecated-gpu-targets -Xfatbin -compress-all")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w")

SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -msse -Xcompiler -msse2 -Xcompiler -msse3")

list(APPEND CMAKE_CUDA_FLAGS "--threads 4")
