execute_process(
    COMMAND ${NVCC_CMD} -Xcompiler "-dumpfullversion -dumpversion"
    OUTPUT_VARIABLE CUDA_CCVER
    ERROR_QUIET
)
execute_process(
    COMMAND ${NVCC_CMD} -Xcompiler "\"-dumpfullversion\" \"-dumpversion\" "
    OUTPUT_VARIABLE CUDA_CCVER
    ERROR_QUIET
)
