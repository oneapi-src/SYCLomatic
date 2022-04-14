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
  if(UNIX)
    set(LIBCURL ${CMAKE_SOURCE_DIR}/../clang/lib/C2S/libcurl/lib/linux/libcurl.a)
  else()
    set(LIBCURL ${CMAKE_SOURCE_DIR}/../clang/lib/C2S/libcurl/lib/win/libcurl_a.lib)
  endif()

  target_link_libraries(c2s
    PRIVATE
    C2S
    ${LIBCURL})

  add_clang_symlink(dpct c2s)

  if(UNIX)
    set(c2s_vars_script vars.sh)
    set(c2s_syscheck_script sys_check.sh)
    set(c2s_modulefiles_script dpct)
  else()
    set(c2s_vars_script vars.bat)
  endif()

  install(FILES ${c2s_vars_script}
          COMPONENT c2s-vars
          PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
          DESTINATION ./env)
  if(UNIX)
    install(FILES ${c2s_syscheck_script}
            COMPONENT c2s-syscheck
            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
            DESTINATION ./sys_check)
  endif()
  install(FILES ${c2s_modulefiles_script}
          COMPONENT c2s-modulefiles
          PERMISSIONS OWNER_READ GROUP_READ WORLD_READ
          DESTINATION ./modulefiles)

  if (NOT CMAKE_CONFIGURATION_TYPES)
    add_llvm_install_targets(install-c2s-vars COMPONENT c2s-vars)
  endif()
  if (NOT CMAKE_CONFIGURATION_TYPES)
    add_llvm_install_targets(install-c2s-syscheck COMPONENT c2s-syscheck)
  endif()
  if (NOT CMAKE_CONFIGURATION_TYPES)
    add_llvm_install_targets(install-c2s-modulefiles COMPONENT c2s-modulefiles)
  endif()
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
      install-c2s-vars
      install-c2s-syscheck
      install-c2s-opt-rules
      )
  else()
    list(APPEND C2S_RUN
      install-c2s-intercept-build
      install-c2s-headers
      install-clang-resource-headers
      install-c2s
      install-c2s-vars
      install-c2s-syscheck
      install-c2s-modulefiles
      install-c2s-opt-rules
      )
  endif()
endmacro()
