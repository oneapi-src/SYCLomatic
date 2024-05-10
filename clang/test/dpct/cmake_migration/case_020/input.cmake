set(SOURCES foo.cu ${CMAKE_SOURCE_DIR}/foo/bar.cu)
list(APPEND SOURCES foo.dp.cpp)
list(append my_libs ${CUDA_LIBRARIES})
list(APPEND FOO_LIBRARIES cuda)
