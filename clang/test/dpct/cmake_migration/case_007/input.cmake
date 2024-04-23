cuda_compile_fatbin(CUDA_FATBINS ${CUDA_FATBIN_SOURCE})

cuda_compile_fatbin(FATBINS main.cu OPTIONS -arch=sm80)

cuda_compile_fatbin(${TARGET} ${CMAKE_SOURCE_DIR}/foo/bar/util.cu)
