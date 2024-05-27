set(NVCC_CMD ${CMAKE_CUDA_COMPILER} .c)
if (NOT CMAKE_CUDA_HOST_COMPILER STREQUAL "")
    set(NVCC_CMD ${NVCC_CMD} -ccbin ${CMAKE_CUDA_HOST_COMPILER})
endif()
set(CMAKE_CUDA_ARCHITECTURES "")
set(CMAKE_CUDA_FLAGS "")

add_custom_command(
    COMMAND ${CMAKE_COMMAND} -E echo \"\#endif // define ${HEADER_FILE_DEFINE}\" >> ${OUTPUT_HEADER_FILE}
    DEPENDS ${spv_file} xxd
    COMMENT "Converting to hpp: ${FILE_NAME} ${CMAKE_BINARY_DIR}/bin/$<CONFIG>/xxd"
    )
set(NVCC_CMD ${CMAKE_CUDA_COMPILER} .c)
string(REGEX REPLACE "^.* version ([0-9.]*).*$" "\\1" CUDA_CCVER ${CUDA_CCFULLVER})
