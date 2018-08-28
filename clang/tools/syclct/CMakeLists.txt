set(LLVM_LINK_COMPONENTS
  support
  )

add_clang_tool(syclct
  SyclCT.cpp
  )

include_directories(
  # For the CUDA-Toolchain and its CudaInstallationDetector
  ../../../clang/lib/Driver/
  )

target_link_libraries(syclct
  PRIVATE
  SyclCT
  )
