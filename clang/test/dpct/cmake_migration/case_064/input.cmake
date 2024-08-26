find_library(CUDA_LIB cuda PATH ${prefix_path})

find_library(CUBLAS_LIB cublas)

find_library(NV_TOOLS_LIBRARIES NAMES nvToolsExt PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
