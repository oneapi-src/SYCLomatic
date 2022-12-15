set(LLVM_LINK_COMPONENTS
  Option
  Support
  )

set(CONT_FILES
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/atomic.hpp.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/device.hpp.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/image.hpp.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/memory.hpp.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/util.hpp.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/blas_utils.hpp.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dnnl_utils.hpp.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpct.hpp.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpl_utils.hpp.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/kernel.hpp.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/rng_utils.hpp.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/lib_common_utils.hpp.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/ccl_utils.hpp.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/fft_utils.hpp.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/lapack_utils.hpp.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpl_extras/algorithm.h.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpl_extras/functional.h.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpl_extras/iterators.h.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpl_extras/memory.h.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpl_extras/numeric.h.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpl_extras/vector.h.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpl_extras/dpcpp_extensions.h.inc
  )

set(PROCESS_FILES_OUTPUT
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dpl_extras/algorithm.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dpl_extras/dpcpp_extensions.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dpl_extras/functional.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dpl_extras/iterators.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dpl_extras/memory.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dpl_extras/numeric.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dpl_extras/vector.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dpl_extras/algorithm.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dpl_extras/dpcpp_extensions.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dpl_extras/functional.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dpl_extras/iterators.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dpl_extras/memory.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dpl_extras/numeric.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dpl_extras/vector.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/atomic.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/blas_utils.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dnnl_utils.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/device.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dpct.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dpl_utils.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/image.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/kernel.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/memory.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/util.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/rng_utils.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/lib_common_utils.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/ccl_utils.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/fft_utils.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/lapack_utils.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/atomic.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/blas_utils.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dnnl_utils.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/device.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dpct.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dpl_utils.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/image.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/kernel.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/memory.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/util.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/rng_utils.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/lib_common_utils.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/ccl_utils.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/fft_utils.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/lapack_utils.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/HelperFeatureEnum.inc
)

add_custom_command(
  OUTPUT Confusables.inc
  COMMAND make_confusable_table ${CMAKE_CURRENT_SOURCE_DIR}/ConfusableTable/confusables.txt ${CMAKE_CURRENT_BINARY_DIR}/Confusables.inc
  DEPENDS make_confusable_table ConfusableTable/confusables.txt
  )

add_custom_command(
  DEPENDS ${CONT_FILES} ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/HelperFileAndFeatureNames.inc ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/processFiles.py
  OUTPUT  ${PROCESS_FILES_OUTPUT}
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/processFiles.py --build-dir ${CMAKE_BINARY_DIR}
  )

add_custom_target(genconfusable 
  DEPENDS Confusables.inc
  )

add_custom_target(dpct_helper_headers_and_inc
  DEPENDS ${PROCESS_FILES_OUTPUT}
  )

include_directories(
  # For the CUDA-Toolchain and its CudaInstallationDetector
  ../../../clang/lib/Driver/
  )

add_subdirectory(ConfusableTable)

build_lib_dpct()
