set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ")
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
if(UNIX)
  set(LIBCURL ${CMAKE_SOURCE_DIR}/../clang/lib/SyclCT/libcurl/lib/linux/libcurl.a)
else()
  set(LIBCURL ${CMAKE_SOURCE_DIR}/../clang/lib/SyclCT/libcurl/lib/win/libcurl_a.lib)
endif()

target_link_libraries(syclct
  PRIVATE
  SyclCT
  ${LIBCURL}
  )
#
if(UNIX)
set(syclct_vars_script
  vars.sh
  )
set(syclct_syscheck_script
  sys_check.sh
)
else()
set(syclct_vars_script
  vars.bat
  )

set(syclct_syscheck_script
sys_check.bat
  )
endif()

install(
  FILES ${syclct_vars_script}
  COMPONENT syclct-vars
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  DESTINATION ./env)


install(
  FILES ${syclct_syscheck_script}
  COMPONENT syclct-syscheck
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  DESTINATION ./env)

if (NOT CMAKE_CONFIGURATION_TYPES) # don't add this for IDE's.
  add_llvm_install_targets(install-syclct-vars
                           COMPONENT syclct-vars)
endif()

if (NOT CMAKE_CONFIGURATION_TYPES) # don't add this for IDE's.
  add_llvm_install_targets(install-syclct-syscheck
                           COMPONENT syclct-syscheck)
endif()

