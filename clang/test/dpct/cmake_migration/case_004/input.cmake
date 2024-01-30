target_link_libraries(main PRIVATE cxxopts fmt::fmt -lcublas)

target_link_libraries(foo3 PUBLIC ${CUDA_LIBRARIES} ${TCNN_LIBRARIES} fmt)

target_link_libraries(transformer_engine PUBLIC 
   CUDA::cublas
   CUDA::cuda_driver
   CUDA::cudart
   CUDA::nvrtc
   CUDA::nvToolsExt
   cudnn)
                     
target_link_libraries(${target} PRIVATE
         -static-libgcc
         -static-libstdc++
         )
                     
target_link_libraries(${PROJECT_NAME}
   libnvinfer.so
   libnvonnxparser.so
)
target_link_libraries(nlm_cuda  ${OpenCV_LIBS} stdc++ stdc++fs)
