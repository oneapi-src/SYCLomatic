cuda_add_executable(cudasift ${cuda_sources} ${sources} OPTIONS -arch=sm_35)
CUDA_ADD_EXECUTABLE(cudasift ${cuda_sources} ${sources} OPTIONS -arch=sm_35)

cuda_add_executable(cudasift src/a.cpp src/main.cpp src/utils.cpp src/b.cpp)

cuda_add_executable(${example}_cuda
                ${CMAKE_CURRENT_BINARY_DIR}/${example}.cu)

gmx_cuda_add_library(foo ${LIBFOO_SOURCES})
