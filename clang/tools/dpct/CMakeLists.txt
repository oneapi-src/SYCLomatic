if(UNIX)
  check_cxx_compiler_flag("-static-intel" CXX_SUPPORTS_STATIC_INTEL_FLAG)
  if( CXX_SUPPORTS_STATIC_INTEL_FLAG )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-intel")
  endif()
endif()

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ")
set(LLVM_LINK_COMPONENTS
  support
  )

add_clang_tool(dpct
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

target_link_libraries(dpct
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
set(dpct_modulefiles_script
  dpct
  )
else()
set(dpct_vars_script
  vars.bat
  )

endif()

install(
  FILES ${dpct_vars_script}
  COMPONENT dpct-vars
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  DESTINATION ./env)

if(UNIX)
install(
  FILES ${dpct_syscheck_script}
  COMPONENT dpct-syscheck
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  DESTINATION ./sys_check)
endif()

install(
  FILES ${dpct_modulefiles_script}
  COMPONENT dpct-modulefiles
  PERMISSIONS OWNER_READ GROUP_READ WORLD_READ
  DESTINATION ./modulefiles)

if (NOT CMAKE_CONFIGURATION_TYPES) # don't add this for IDE's.
  add_llvm_install_targets(install-dpct-vars
                           COMPONENT dpct-vars)
endif()

if (NOT CMAKE_CONFIGURATION_TYPES) # don't add this for IDE's.
  add_llvm_install_targets(install-dpct-syscheck
                           COMPONENT dpct-syscheck)
endif()

if (NOT CMAKE_CONFIGURATION_TYPES) # don't add this for IDE's.
  add_llvm_install_targets(install-dpct-modulefiles
                           COMPONENT dpct-modulefiles)
endif()

add_subdirectory(DpctOptRules)
