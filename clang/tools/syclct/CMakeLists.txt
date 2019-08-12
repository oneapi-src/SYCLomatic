set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ")
set(LLVM_LINK_COMPONENTS
  support
  )

add_clang_tool(syclct
  DPCT.cpp
  )
include_directories(
  # For the CUDA-Toolchain and its CudaInstallationDetector
  ../../../clang/lib/Driver/
  )
if(UNIX)
  set(LIBCURL ${CMAKE_SOURCE_DIR}/../clang/lib/DPCT/libcurl/lib/linux/libcurl.a)
else()
  set(LIBCURL ${CMAKE_SOURCE_DIR}/../clang/lib/DPCT/libcurl/lib/win/libcurl_a.lib)
endif()

target_link_libraries(syclct
  PRIVATE
  DPCT
  ${LIBCURL}
  )
#
if(UNIX)
set(dpct_vars_script
  vars.sh
  )
set(dpct_syscheck_script
  sys_check.sh
)
else()
set(dpct_vars_script
  vars.bat
  )

set(dpct_syscheck_script
sys_check.bat
  )
endif()

install(
  FILES ${dpct_vars_script}
  COMPONENT dpct-vars
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  DESTINATION ./env)


install(
  FILES ${dpct_syscheck_script}
  COMPONENT dpct-syscheck
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  DESTINATION ./sys_check)

if (NOT CMAKE_CONFIGURATION_TYPES) # don't add this for IDE's.
  add_llvm_install_targets(install-dpct-vars
                           COMPONENT dpct-vars)
endif()

if (NOT CMAKE_CONFIGURATION_TYPES) # don't add this for IDE's.
  add_llvm_install_targets(install-dpct-syscheck
                           COMPONENT dpct-syscheck)
endif()

