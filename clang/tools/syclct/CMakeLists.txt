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

set(syclct_syscheck_script
  sys_check.sh
)

install(
  FILES ${syclct_startup_script}
  COMPONENT syclct-startup
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  DESTINATION ./)


install(
  FILES ${syclct_syscheck_script}
  COMPONENT syclct-syscheck
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  DESTINATION ./sys_check/)

if (NOT CMAKE_CONFIGURATION_TYPES) # don't add this for IDE's.
  add_llvm_install_targets(install-syclct-startup
                           COMPONENT syclct-startup)
endif()

if (NOT CMAKE_CONFIGURATION_TYPES) # don't add this for IDE's.
  add_llvm_install_targets(install-syclct-syscheck
                           COMPONENT syclct-syscheck)
endif()

