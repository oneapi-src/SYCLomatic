cuda_compile_fatbin(CUDA_FATBINS ${CUDA_FATBIN_SOURCE})

cuda_compile_fatbin(FATBINS main.cu OPTIONS -arch=sm80)

cuda_compile_fatbin(${TARGET} ${CMAKE_SOURCE_DIR}/foo/bar/util.cu)

cuda_compile_fatbin(FATBINS main.cu OPTIONS -arch=sm80 -DNV_KERNEL)

cuda_compile_fatbin(${TARGET} main.cu OPTIONS --profile -DCUDA -arch=sm80 --use_fast_math -DNV_KERNEL)
