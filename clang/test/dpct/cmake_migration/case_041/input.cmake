target_link_directories(CUDA::toolkit INTERFACE "${CUDAToolkit_LIBRARY_DIR}")

target_link_directories( CUDA::toolkit INTERFACE "${CUDAToolkit_LIBRARY_DIR}")

target_link_directories( CUDA::toolkit INTERFACE "${CUDAToolkit_LIBRARY_DIR}" )

target_link_options(quda PUBLIC $<$<CUDA_COMPILER_ID:Clang>: --cuda-path=${CUDAToolkit_TARGET_DIR}>)

target_link_options( quda PUBLIC $<$<CUDA_COMPILER_ID:Clang>: --cuda-path=${CUDAToolkit_TARGET_DIR}>)

target_link_options(quda PUBLIC $<$<CUDA_COMPILER_ID:Clang>: --cuda-path=${CUDAToolkit_TARGET_DIR}> )

target_link_options(nvcv_util_compat
PUBLIC
    -static-libstdc++
    -static-libgcc
    -Wl,--wrap=__libc_start_main
    -Wl,-u__cxa_thread_atexit_impl
    ${linkcompat}
    -Wl,--push-state,--no-as-needed
    ${CMAKE_CURRENT_SOURCE_DIR}/stubs/libdl-2.17_stub.so
    ${CMAKE_CURRENT_SOURCE_DIR}/stubs/librt-2.17_stub.so
    ${CMAKE_CURRENT_SOURCE_DIR}/stubs/libpthread-2.17_stub.so
    -Wl,--pop-state
)

target_link_options(opts
PRIVATE
    ${CMAKE_CUDA_RUNTIME_LIBRARY_LINK_OPTIONS_SHARED}
    ${CMAKE_CUDA_RUNTIME_LIBRARY_LINK_OPTIONS_STATIC}
)
