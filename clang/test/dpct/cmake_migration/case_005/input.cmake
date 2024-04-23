set_property(SOURCE cuda_file.cu)

set_property(TARGET a
   SOURCE cuda_file.cu)

set_property(SOURCE cuda_file.cu cuda_file2.cu cuda_file3.cu)

set_property(
   SOURCE cuda_file.cu
   DIRECTORY ${CMAKE_SOURCE_DIR}
   APPEND
   PROPERTY COMPILE_DEFINITIONS ${BACKWARD_DEFINITIONS})
