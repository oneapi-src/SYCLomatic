target_source(ATarget PRIVATE src1.cu src2.cu)

target_include_directories(nvc_get_devices PRIVATE ${CUDA_INCLUDE_DIRS})

target_include_directories(${lib_name} SYSTEM INTERFACE "${CUDAToolkit_INCLUDE_DIRS}")

# TransformerEngine/transformer_engine/cmake/FindCUDNN.cmake
target_include_directories(target $<BUILD_INTERFACE:${CUDNN_INCLUDE_DIR}>)

target_include_directories(nvshmem_device_lib INTERFACE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_include_directories(quda SYSTEM PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${CUDAToolkit_INCLUDE_DIRS}>)

target_include_directories(quda SYSTEM PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${CUDAToolkit_MATH_INCLUDE_DIR}>)

target_include_directories(quda_cpp SYSTEM PUBLIC ${CUDAToolkit_INCLUDE_DIRS} ${CUDAToolkit_MATH_INCLUDE_DIR})
