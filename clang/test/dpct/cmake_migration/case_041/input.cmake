target_link_directories(CUDA::toolkit INTERFACE "${CUDAToolkit_LIBRARY_DIR}")

target_link_directories( CUDA::toolkit INTERFACE "${CUDAToolkit_LIBRARY_DIR}")

target_link_directories( CUDA::toolkit INTERFACE "${CUDAToolkit_LIBRARY_DIR}" )

target_link_options(quda PUBLIC $<$<CUDA_COMPILER_ID:Clang>: --cuda-path=${CUDAToolkit_TARGET_DIR}>)

target_link_options( quda PUBLIC $<$<CUDA_COMPILER_ID:Clang>: --cuda-path=${CUDAToolkit_TARGET_DIR}>)

target_link_options(quda PUBLIC $<$<CUDA_COMPILER_ID:Clang>: --cuda-path=${CUDAToolkit_TARGET_DIR}> )
