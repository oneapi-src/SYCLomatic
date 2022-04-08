set(c2s_rt_files
  ${CMAKE_BINARY_DIR}/tools/clang/runtime/c2s-rt/include/c2s.hpp
  ${CMAKE_BINARY_DIR}/tools/clang/runtime/c2s-rt/include/atomic.hpp
  ${CMAKE_BINARY_DIR}/tools/clang/runtime/c2s-rt/include/device.hpp
  ${CMAKE_BINARY_DIR}/tools/clang/runtime/c2s-rt/include/memory.hpp
  ${CMAKE_BINARY_DIR}/tools/clang/runtime/c2s-rt/include/kernel.hpp
  ${CMAKE_BINARY_DIR}/tools/clang/runtime/c2s-rt/include/image.hpp
  ${CMAKE_BINARY_DIR}/tools/clang/runtime/c2s-rt/include/util.hpp
  ${CMAKE_BINARY_DIR}/tools/clang/runtime/c2s-rt/include/blas_utils.hpp
  ${CMAKE_BINARY_DIR}/tools/clang/runtime/c2s-rt/include/dpl_utils.hpp
  ${CMAKE_BINARY_DIR}/tools/clang/runtime/c2s-rt/include/rng_utils.hpp
  ${CMAKE_BINARY_DIR}/tools/clang/runtime/c2s-rt/include/lib_common_utils.hpp
)

set(c2s_rt_dpstd_files
  ${CMAKE_BINARY_DIR}/tools/clang/runtime/c2s-rt/include/dpl_extras/functional.h
  ${CMAKE_BINARY_DIR}/tools/clang/runtime/c2s-rt/include/dpl_extras/memory.h
  ${CMAKE_BINARY_DIR}/tools/clang/runtime/c2s-rt/include/dpl_extras/iterators.h
  ${CMAKE_BINARY_DIR}/tools/clang/runtime/c2s-rt/include/dpl_extras/algorithm.h
  ${CMAKE_BINARY_DIR}/tools/clang/runtime/c2s-rt/include/dpl_extras/numeric.h
  ${CMAKE_BINARY_DIR}/tools/clang/runtime/c2s-rt/include/dpl_extras/vector.h
  ${CMAKE_BINARY_DIR}/tools/clang/runtime/c2s-rt/include/dpl_extras/dpcpp_extensions.h
)

set(c2s_rt_backward_compatibility_files
  ${CMAKE_SOURCE_DIR}/../clang/runtime/c2s-rt/include/backward_compatibility/dpct.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/c2s-rt/include/backward_compatibility/atomic.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/c2s-rt/include/backward_compatibility/device.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/c2s-rt/include/backward_compatibility/memory.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/c2s-rt/include/backward_compatibility/kernel.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/c2s-rt/include/backward_compatibility/image.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/c2s-rt/include/backward_compatibility/util.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/c2s-rt/include/backward_compatibility/blas_utils.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/c2s-rt/include/backward_compatibility/dpl_utils.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/c2s-rt/include/backward_compatibility/rng_utils.hpp
)

set(c2s_rt_dpstd_backward_compatibility_files
  ${CMAKE_SOURCE_DIR}/../clang/runtime/c2s-rt/include/backward_compatibility/dpl_extras/functional.h
  ${CMAKE_SOURCE_DIR}/../clang/runtime/c2s-rt/include/backward_compatibility/dpl_extras/memory.h
  ${CMAKE_SOURCE_DIR}/../clang/runtime/c2s-rt/include/backward_compatibility/dpl_extras/iterators.h
  ${CMAKE_SOURCE_DIR}/../clang/runtime/c2s-rt/include/backward_compatibility/dpl_extras/algorithm.h
  ${CMAKE_SOURCE_DIR}/../clang/runtime/c2s-rt/include/backward_compatibility/dpl_extras/numeric.h
  ${CMAKE_SOURCE_DIR}/../clang/runtime/c2s-rt/include/backward_compatibility/dpl_extras/vector.h
  ${CMAKE_SOURCE_DIR}/../clang/runtime/c2s-rt/include/backward_compatibility/dpl_extras/dpcpp_extensions.h
)

add_custom_target(c2s-headers ALL DEPENDS c2s_helper_headers_and_inc)

install(
  FILES ${c2s_rt_files}
  COMPONENT c2s-headers
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
  DESTINATION include/c2s)

install(
  FILES ${c2s_rt_dpstd_files}
  COMPONENT c2s-headers
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
  DESTINATION include/c2s/dpl_extras)

install(
  FILES ${c2s_rt_backward_compatibility_files}
  COMPONENT c2s-headers
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
  DESTINATION include/dpct)

install(
  FILES ${c2s_rt_dpstd_backward_compatibility_files}
  COMPONENT c2s-headers
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
  DESTINATION include/dpct/dpl_extras)

if (NOT CMAKE_CONFIGURATION_TYPES) # don't add this for IDE's.
  add_llvm_install_targets(install-c2s-headers
                           DEPENDS c2s-headers
                           COMPONENT c2s-headers)
endif()