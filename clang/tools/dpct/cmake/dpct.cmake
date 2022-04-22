macro(set_dpct_package)
  if(WIN32)
    list(APPEND DPCT_RUN
      install-dpct-headers
      install-clang-resource-headers
      install-dpct-binary
      install-dpct-opt-rules
      )
  else()
    list(APPEND DPCT_RUN
      install-dpct-intercept-build
      install-dpct-headers
      install-clang-resource-headers
      install-dpct-binary
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
endmacro()
