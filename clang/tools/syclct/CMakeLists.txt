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

#

set(syclct_startup_script
  startup.sh
  )
install(
  FILES ${syclct_startup_script}
  COMPONENT syclct-startup
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
  DESTINATION ./)

if (NOT CMAKE_CONFIGURATION_TYPES) # don't add this for IDE's.
  add_llvm_install_targets(install-syclct-startup
                           COMPONENT syclct-startup)
endif()

