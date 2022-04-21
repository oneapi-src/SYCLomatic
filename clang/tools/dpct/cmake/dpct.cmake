macro(set_dpct_package)
  if(WIN32)
    list(APPEND DPCT_RUN
      install-dpct-headers
      install-clang-resource-headers
      install-dpct-binary
      install-dpct-vars
      install-dpct-syscheck
      install-dpct-opt-rules
      )
  else()
    list(APPEND DPCT_RUN
      install-dpct-intercept-build
      install-dpct-headers
      install-clang-resource-headers
      install-dpct-binary
      install-dpct-vars
      install-dpct-syscheck
      install-dpct-modulefiles
      install-dpct-opt-rules
      )
  endif()
endmacro()

macro(install_dpct)
  if(UNIX)
    set(LIBCURL ${CMAKE_SOURCE_DIR}/../clang/lib/DPCT/libcurl/lib/linux/libcurl.a)
  else()
    set(LIBCURL ${CMAKE_SOURCE_DIR}/../clang/lib/DPCT/libcurl/lib/win/libcurl_a.lib)
  endif()

  target_link_libraries(dpct-binary
    PRIVATE
    DPCT
    ${LIBCURL}
    )

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
endmacro()
