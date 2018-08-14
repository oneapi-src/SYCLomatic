set(LLVM_LINK_COMPONENTS
  support
  )

add_clang_tool(cu2sycl
  Cu2Sycl.cpp
  )

include_directories(
  # For the CUDA-Toolchain and its CudaInstallationDetector
  ../../../clang/lib/Driver/
  )

target_link_libraries(cu2sycl
  PRIVATE
  Cu2Sycl
  )
