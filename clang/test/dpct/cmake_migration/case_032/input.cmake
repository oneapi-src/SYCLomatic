link_libraries(main PRIVATE cxxopts fmt::fmt -lcublas)

link_libraries(main PUBLIC cuda cublas)

link_libraries(foo3 PUBLIC ${CUDA_LIBRARIES} ${TCNN_LIBRARIES} fmt)

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
