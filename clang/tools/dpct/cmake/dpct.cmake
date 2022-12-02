macro(set_dpct_package)
  if(WIN32)
    list(APPEND DPCT_RUN
      install-dpct-pattern-rewriter
      install-dpct-headers
      install-clang-resource-headers
      install-dpct-binary
      install-dpct-opt-rules
      )
  else()
    list(APPEND DPCT_RUN
      install-dpct-intercept-build
      install-dpct-pattern-rewriter
      install-dpct-headers
      install-clang-resource-headers
      install-dpct-binary
      install-dpct-autocomplete
      install-dpct-opt-rules
      )
  endif()
endmacro()

macro(install_dpct)
  target_link_libraries(dpct-binary
    PRIVATE
    DPCT
    )

  add_clang_symlink(c2s dpct-binary)
  if(UNIX)
    set(dpct_autocomplete_script bash-autocomplete.sh)
  endif()

  if(UNIX)
    install(
      FILES ${dpct_autocomplete_script}
      COMPONENT dpct-autocomplete
      PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
      DESTINATION ./env)
    if (NOT CMAKE_CONFIGURATION_TYPES)
      add_llvm_install_targets(install-dpct-autocomplete
                               COMPONENT dpct-autocomplete)
    endif()
  endif()
endmacro()
