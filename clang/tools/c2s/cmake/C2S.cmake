macro(install_libscanbuild_libs)
  install(PROGRAMS lib/libscanbuild/${lib}
          DESTINATION lib/libscanbuild
          COMPONENT c2s-intercept-build)
endmacro()

macro(install_ear)
  install(TARGETS ear
          LIBRARY
          DESTINATION lib/libear
          COMPONENT c2s-intercept-build)
endmacro()

macro(install_intercept_stub)
  install(TARGETS intercept-stub
          DESTINATION lib/libear
          COMPONENT c2s-intercept-build)
endmacro()


macro(install_init_py)
  install(PROGRAMS lib/libear/__init__.py
          DESTINATION lib/libear
          COMPONENT c2s-intercept-build)
endmacro()

macro(install_intercept_build)
  install(PROGRAMS bin/intercept-build
          DESTINATION bin
          COMPONENT c2s-intercept-build)
endmacro()

macro(install_c2s)
  target_link_libraries(c2s
    PRIVATE
    C2S)

  add_clang_symlink(dpct c2s)
endmacro()

macro(install_c2s_rule_files)
  install(FILES ${c2s_opt_rule_files}
          COMPONENT c2s-opt-rules
          PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
          DESTINATION ./extensions/opt_rules)
endmacro()

macro(set_header_install_dir)
  set(header_install_dir lib${LLVM_LIBDIR_SUFFIX}/clang/${CLANG_VERSION}/include)
endmacro()

macro(set_c2s_package)
  if(WIN32)
    list(APPEND C2S_RUN
      install-c2s-headers
      install-clang-resource-headers
      install-c2s
      install-c2s-opt-rules
      )
  else()
    list(APPEND C2S_RUN
      install-c2s-intercept-build
      install-c2s-headers
      install-clang-resource-headers
      install-c2s
      install-c2s-opt-rules
      )
  endif()
endmacro()
