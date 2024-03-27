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

set(LIBS ${LIBS}
    ${CUDA_LIBRARIES}
    ${CUDA_cudart_static_LIBRARY}
    ${CUDA_cusolver_LIBRARY}
    ${CUDA_nppc_LIBRARY}
    ${CUDA_nppif_LIBRARY}
    ${CUDA_nppim_LIBRARY}
    ${CUDA_npps_LIBRARY}
    ${CUDA_nvcuvenc_LIBRARY}
)

target_link_libraries(${target}
    ${CUDA_cudadevrt_LIBRARY}
    ${CUDA_CUFFT_LIBRARIES}
    ${CUDA_curand_LIBRARY}
    ${CUDA_nppial_LIBRARY}
    ${CUDA_nppicc_LIBRARY}
    ${CUDA_nppist_LIBRARY}
    ${CUDA_nppisu_LIBRARY}
    ${CUDA_nvcuvid_LIBRARY}
    ${CUDA_nvToolsExt_LIBRARY}
)

link_libraries(PUBLIC
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_cusparse_LIBRARY}
    ${CUDA_cupti_LIBRARY}
    ${CUDA_CUT_LIBRARY}
    ${CUDA_npp_LIBRARY}
    ${CUDA_nppi_LIBRARY}
    ${CUDA_nppicom_LIBRARY}
    ${CUDA_nppidei_LIBRARY}
    ${CUDA_nppig_LIBRARY}
    ${CUDA_nppitc_LIBRARY}
)

set(LIBS ${LIBS}
    CUDA::cudla
    CUDA::cuFile
    CUDA::cuFile_static
    CUDA::cuFile_rdma
    CUDA::cupti
    CUDA::cufft
    CUDA::nvperf_host
    CUDA::nvperf_target
    CUDA::pcsamplingutil
    CUDA::nppc
    CUDA::nppial
    CUDA::nppicc
    CUDA::nppicom
    CUDA::cufftw
    CUDA::nppidei
    CUDA::nppif
    CUDA::nppig
    CUDA::cusparse
    CUDA::nppim
    CUDA::nppist
)

target_link_libraries(${target}
    CUDA::cuFile_rdma_static
    CUDA::cupti_static
    CUDA::nvperf_host_static
    CUDA::cusparse_static
    CUDA::nppc_static
    CUDA::nppial_static
    CUDA::cufft_static
    CUDA::nppicc_static
    CUDA::nppicom_static
    CUDA::nppidei_static
    CUDA::nppif_static
    CUDA::nppig_static
    CUDA::cufft_static_nocallback
    CUDA::nppim_static
    CUDA::nppist_static
    CUDA::nppisu
    CUDA::nppitc
    CUDA::npps
    CUDA::cufftw_static
    CUDA::nvblas
    CUDA::nvgraph
)

link_libraries(PUBLIC
    CUDA::curand
    CUDA::nppisu_static
    CUDA::nppitc_static
    CUDA::npps_static
    CUDA::nvgraph_static
    CUDA::nvjpeg
    CUDA::nvjpeg_static
    CUDA::nvptxcompiler_static
    CUDA::nvrtc_builtins
    CUDA::curand_static
    CUDA::nvrtc_static
    CUDA::nvrtc_builtins_static
    CUDA::nvJitLink
    CUDA::nvJitLink_static
    CUDA::cusolver
    CUDA::nvml
    CUDA::nvtx3
    CUDA::OpenCL
    CUDA::cusolver_static
)
