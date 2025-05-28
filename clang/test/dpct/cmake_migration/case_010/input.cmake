file(GLOB REL_CUDA_SRC "file1.cu" "${CMAKE_SOURCE_DIR}/src/*.cu")
file(GLOB SRCS *.cc *.cu *.cuh)
file(GLOB_RECURSE DECODE_KERNELS_SRCS
     ${PROJECT_SOURCE_DIR}/src/generated/d*decode_head*.cu)
file(GLOB_RECURSE PREFILL_KERNELS_SRCS
     ${PROJECT_SOURCE_DIR}/src/generated/*prefill_head*.cu)
