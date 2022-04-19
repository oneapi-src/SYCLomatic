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

add_clang_tool(c2s
  C2S.cpp
  )
include_directories(
  # For the CUDA-Toolchain and its CudaInstallationDetector
  ../../../clang/lib/Driver/
  )
if(UNIX)
  set(LIBCURL ${CMAKE_SOURCE_DIR}/../clang/lib/C2S/libcurl/lib/linux/libcurl.a)
else()
  set(LIBCURL ${CMAKE_SOURCE_DIR}/../clang/lib/C2S/libcurl/lib/win/libcurl_a.lib)
endif()

target_link_libraries(c2s
  PRIVATE
  C2S
  ${LIBCURL}
  )

add_clang_symlink(dpct c2s)

if(UNIX)
set(c2s_vars_script
  vars.sh
  )
set(c2s_syscheck_script
  sys_check.sh
)
set(c2s_modulefiles_script
  dpct
  )
else()
set(c2s_vars_script
  vars.bat
  )

endif()

if(INTEL_DEPLOY_UNIFIED_LAYOUT)
  install(
    FILES ${c2s_vars_script}
    COMPONENT c2s-vars
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    DESTINATION etc/c2s)

  if(UNIX)
  install(
    FILES ${c2s_syscheck_script}
    COMPONENT c2s-syscheck
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    DESTINATION etc/c2s)
  endif()

  install(
    FILES ${c2s_modulefiles_script}
    COMPONENT c2s-modulefiles
    PERMISSIONS OWNER_READ GROUP_READ WORLD_READ
    DESTINATION etc/c2s/modulefiles)
else()
  install(
    FILES ${c2s_vars_script}
    COMPONENT c2s-vars
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    DESTINATION ./env)

  if(UNIX)
  install(
    FILES ${c2s_syscheck_script}
    COMPONENT c2s-syscheck
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    DESTINATION ./sys_check)
  endif()

  install(
    FILES ${c2s_modulefiles_script}
    COMPONENT c2s-modulefiles
    PERMISSIONS OWNER_READ GROUP_READ WORLD_READ
    DESTINATION ./modulefiles)
endif()

if (NOT CMAKE_CONFIGURATION_TYPES) # don't add this for IDE's.
  add_llvm_install_targets(install-c2s-vars
                           COMPONENT c2s-vars)
endif()

if (NOT CMAKE_CONFIGURATION_TYPES) # don't add this for IDE's.
  add_llvm_install_targets(install-c2s-syscheck
                           COMPONENT c2s-syscheck)
endif()

if (NOT CMAKE_CONFIGURATION_TYPES) # don't add this for IDE's.
  add_llvm_install_targets(install-c2s-modulefiles
                           COMPONENT c2s-modulefiles)
endif()

add_subdirectory(C2SOptRules)
