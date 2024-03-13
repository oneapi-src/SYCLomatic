link_libraries(main PRIVATE cxxopts fmt::fmt -lcublas)

link_libraries(main PUBLIC cuda cublas)

link_libraries(foo3 PUBLIC ${CUDA_LIBRARIES} ${CUDA_cudadevrt_LIBRARY} ${TCNN_LIBRARIES} fmt)

link_libraries(transformer_engine PUBLIC 
   CUDA::cublas
   CUDA::cuda_driver
   CUDA::cudart
   CUDA::nvrtc
   CUDA::nvToolsExt
   cudnn)
                     
link_libraries(${target} PRIVATE
         -static-libgcc
         -static-libstdc++
         )
                     
link_libraries(${PROJECT_NAME}
   libnvinfer.so
   libnvonnxparser.so
)

if (WIN32)
    # As of 12.3.1 CUDA Tookit for Windows does not offer a static cublas library
    set(LLAMA_EXTRA_LIBS ${LLAMA_EXTRA_LIBS} CUDA::cudart_static CUDA::cublas CUDA::cublasLt)
else ()
    set(LLAMA_EXTRA_LIBS ${LLAMA_EXTRA_LIBS} CUDA::cudart_static CUDA::cublas_static CUDA::cublasLt_static)
endif()
set(LLAMA_EXTRA_LIBS ${LLAMA_EXTRA_LIBS} CUDA::cudart CUDA::cublas CUDA::cublasLt)
