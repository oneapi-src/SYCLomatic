find_program(CUDAToolkit_NVCC_EXECUTABLE nvcc ${CUDAToolkit_BIN_DIR})

find_program(CUDAToolkit_NVCC_other_EXE nvcc_other)

find_program(CUDAToolkit_NVCC_EXECUTABLE
    NAMES nvcc nvcc.exe
    PATHS ${CUDAToolkit_BIN_DIR}
    NO_DEFAULT_PATH
    )

find_program(BIN2C bin2c)
