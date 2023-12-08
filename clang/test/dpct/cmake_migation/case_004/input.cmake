target_link_libraries(main PRIVATE cxxopts fmt::fmt -lcublas)

target_link_libraries(tiny-cuda-nn PUBLIC ${CUDA_LIBRARIES} ${TCNN_LIBRARIES} fmt)

target_link_libraries(transformer_engine PUBLIC 
   CUDA::cublas
   CUDA::cuda_driver
   CUDA::cudart
   CUDA::nvrtc
   CUDA::nvToolsExt
   CUDNN::cudnn)
                     
target_link_libraries(${target} PRIVATE
         -static-libgcc
         -static-libstdc++
         )
                     
target_link_libraries(${PROJECT_NAME}
   libnvinfer.so
   libnvonnxparser.so
)