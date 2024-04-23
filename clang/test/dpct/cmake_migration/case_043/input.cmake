include_directories(${CUDA_NVCC_INCLUDE_DIRS})

target_include_directories(my_cuda_program PRIVATE ${CUDA_NVCC_INCLUDE_DIRS})

list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${COMPILE_FLAG}")

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17")
