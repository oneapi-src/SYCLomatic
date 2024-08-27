find_library(CUDA_LIB cuda PATHS ${prefix_path})

find_library(CUDA_LIBS cublas cudnn cufftMp nvshmem cudart_static cufft_static culibos PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib)

find_library(NPP_LIBS nppc_static nppicc_static nppig_static PATHS ENV NPP_PATHS)

find_library(NVJPEG_LIBS nvjpeg2k_static libnvjpeg_static.a nvjpeg PATHS ${NPP_PATHS})

find_library(NV_TOOLS_LIBRARIES NAMES nvToolsExt PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
