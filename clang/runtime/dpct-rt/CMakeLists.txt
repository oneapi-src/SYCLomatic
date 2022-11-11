set(dpct_rt_files
  ${CMAKE_BINARY_DIR}/tools/clang/runtime/dpct-rt/include/dpct.hpp
  ${CMAKE_BINARY_DIR}/tools/clang/runtime/dpct-rt/include/atomic.hpp
  ${CMAKE_BINARY_DIR}/tools/clang/runtime/dpct-rt/include/device.hpp
  ${CMAKE_BINARY_DIR}/tools/clang/runtime/dpct-rt/include/memory.hpp
  ${CMAKE_BINARY_DIR}/tools/clang/runtime/dpct-rt/include/kernel.hpp
  ${CMAKE_BINARY_DIR}/tools/clang/runtime/dpct-rt/include/image.hpp
  ${CMAKE_BINARY_DIR}/tools/clang/runtime/dpct-rt/include/util.hpp
  ${CMAKE_BINARY_DIR}/tools/clang/runtime/dpct-rt/include/blas_utils.hpp
  ${CMAKE_BINARY_DIR}/tools/clang/runtime/dpct-rt/include/dnnl_utils.hpp
  ${CMAKE_BINARY_DIR}/tools/clang/runtime/dpct-rt/include/dpl_utils.hpp
  ${CMAKE_BINARY_DIR}/tools/clang/runtime/dpct-rt/include/rng_utils.hpp
  ${CMAKE_BINARY_DIR}/tools/clang/runtime/dpct-rt/include/lib_common_utils.hpp
  ${CMAKE_BINARY_DIR}/tools/clang/runtime/dpct-rt/include/ccl_utils.hpp
  ${CMAKE_BINARY_DIR}/tools/clang/runtime/dpct-rt/include/sparse_utils.hpp
  ${CMAKE_BINARY_DIR}/tools/clang/runtime/dpct-rt/include/fft_utils.hpp
)

set(dpct_rt_dpstd_files
  ${CMAKE_BINARY_DIR}/tools/clang/runtime/dpct-rt/include/dpl_extras/functional.h
  ${CMAKE_BINARY_DIR}/tools/clang/runtime/dpct-rt/include/dpl_extras/memory.h
  ${CMAKE_BINARY_DIR}/tools/clang/runtime/dpct-rt/include/dpl_extras/iterators.h
  ${CMAKE_BINARY_DIR}/tools/clang/runtime/dpct-rt/include/dpl_extras/algorithm.h
  ${CMAKE_BINARY_DIR}/tools/clang/runtime/dpct-rt/include/dpl_extras/numeric.h
  ${CMAKE_BINARY_DIR}/tools/clang/runtime/dpct-rt/include/dpl_extras/vector.h
  ${CMAKE_BINARY_DIR}/tools/clang/runtime/dpct-rt/include/dpl_extras/dpcpp_extensions.h
)

add_custom_target(dpct-headers ALL DEPENDS dpct_helper_headers_and_inc)

install(
  FILES ${dpct_rt_files}
  COMPONENT dpct-headers
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
  DESTINATION include/dpct)

install(
  FILES ${dpct_rt_dpstd_files}
  COMPONENT dpct-headers
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
  DESTINATION include/dpct/dpl_extras)

if (NOT CMAKE_CONFIGURATION_TYPES) # don't add this for IDE's.
  add_llvm_install_targets(install-dpct-headers
                           DEPENDS dpct-headers
                           COMPONENT dpct-headers)
endif()
