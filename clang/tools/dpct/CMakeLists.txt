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

add_clang_tool(dpct-binary
  DPCT.cpp NO_INSTALL_RPATH
  )
set_target_properties(dpct-binary PROPERTIES OUTPUT_NAME "dpct")
include_directories(
  # For the CUDA-Toolchain and its CudaInstallationDetector
  ../../../clang/lib/Driver/
  )

target_link_libraries(dpct-binary
  PRIVATE
  DPCT
  )

add_clang_symlink(c2s dpct-binary)

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

if(INTEL_DEPLOY_UNIFIED_LAYOUT)
  install(
    FILES ${dpct_vars_script}
    COMPONENT dpct-vars
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    DESTINATION etc/dpct)
  if(UNIX)
    install(
      FILES ${dpct_syscheck_script}
      COMPONENT dpct-syscheck
      PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
      DESTINATION etc/dpct)
  endif()
  install(
    FILES ${dpct_modulefiles_script}
    COMPONENT dpct-modulefiles
    PERMISSIONS OWNER_READ GROUP_READ WORLD_READ
    DESTINATION etc/dpct/modulefiles)
else()
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
endif()

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

if(UNIX)
  set(dpct_autocomplete_script bash-autocomplete.sh)
endif()

if(UNIX)
  if(INTEL_DEPLOY_UNIFIED_LAYOUT)
    install(
            FILES ${dpct_autocomplete_script}
            COMPONENT dpct-autocomplete
            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
            DESTINATION ./etc/dpct)
  else()
    install(
            FILES ${dpct_autocomplete_script}
            COMPONENT dpct-autocomplete
            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
            DESTINATION ./env)
  endif()
  if (NOT CMAKE_CONFIGURATION_TYPES)
    add_llvm_install_targets(install-dpct-autocomplete
                             COMPONENT dpct-autocomplete)
  endif()
endif()

add_subdirectory(DpctOptRules)
