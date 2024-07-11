set (BinFiles
     "analyze-build"
     "intercept-build"
     "scan-build")

set (LibExecs
     "analyze-c++"
     "analyze-cc"
     "intercept-c++"
     "intercept-cc")

set (LibScanbuild
     "__init__.py"
     "analyze.py"
     "arguments.py"
     "clang.py"
     "compilation.py"
     "intercept.py"
     "report.py"
     "shell.py"
     "parse_buildlog.py")

set (LibScanbuildResources
     "scanview.css"
     "selectable.js"
     "sorttable.js")

# libear is compiled dynamically in build_libear using the specified cc
# compiler.
set (LibEar
     "__init__.py"
     "config.h.in"
     "ear.c")

foreach(BinFile ${BinFiles})
  if ("${BinFile}" STREQUAL "scan-build")
    # Need to rename scan-build to scan-build-py to prevent overwriting
    # scan-build Perl implementation.
    add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/bin/scan-build-py
                       COMMAND ${CMAKE_COMMAND} -E make_directory
                         ${CMAKE_BINARY_DIR}/bin
                       COMMAND ${CMAKE_COMMAND} -E copy
                         ${CMAKE_CURRENT_SOURCE_DIR}/bin/scan-build
                         ${CMAKE_BINARY_DIR}/bin/scan-build-py
                       DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/bin/scan-build)
    install (PROGRAMS "bin/scan-build"
             DESTINATION "${CMAKE_INSTALL_BINDIR}"
             RENAME scan-build-py
             COMPONENT scan-build-py)
    list(APPEND Depends ${CMAKE_BINARY_DIR}/bin/scan-build-py)
  else()
    add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/bin/${BinFile}
                       COMMAND ${CMAKE_COMMAND} -E make_directory
                         ${CMAKE_BINARY_DIR}/bin
                       COMMAND ${CMAKE_COMMAND} -E copy
                         ${CMAKE_CURRENT_SOURCE_DIR}/bin/${BinFile}
                         ${CMAKE_BINARY_DIR}/bin/
                       DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/bin/${BinFile})
    install(PROGRAMS bin/${BinFile}
            DESTINATION "${CMAKE_INSTALL_BINDIR}"
            COMPONENT scan-build-py)
    list(APPEND Depends ${CMAKE_BINARY_DIR}/bin/${BinFile})
  endif()
endforeach()

foreach(lib ${LibExecs})
  add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/libexec/${lib}
                     COMMAND ${CMAKE_COMMAND} -E make_directory
                       ${CMAKE_BINARY_DIR}/libexec
                     COMMAND ${CMAKE_COMMAND} -E copy
                       ${CMAKE_CURRENT_SOURCE_DIR}/libexec/${lib}
                       ${CMAKE_BINARY_DIR}/libexec/
                     DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/libexec/${lib})
  list(APPEND Depends ${CMAKE_BINARY_DIR}/libexec/${lib})
  install(PROGRAMS libexec/${lib}
          DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}"
          COMPONENT scan-build-py)
endforeach()

foreach(lib ${LibScanbuild})
  add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/lib/libscanbuild/${lib}
                     COMMAND ${CMAKE_COMMAND} -E make_directory
                       ${CMAKE_BINARY_DIR}/lib
                     COMMAND ${CMAKE_COMMAND} -E make_directory
                       ${CMAKE_BINARY_DIR}/lib/libscanbuild
                     COMMAND ${CMAKE_COMMAND} -E copy
                       ${CMAKE_CURRENT_SOURCE_DIR}/lib/libscanbuild/${lib}
                       ${CMAKE_BINARY_DIR}/lib/libscanbuild/
                     DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/lib/libscanbuild/${lib})
  list(APPEND Depends ${CMAKE_BINARY_DIR}/lib/libscanbuild/${lib})
  install(FILES lib/libscanbuild/${lib}
          DESTINATION lib${CLANG_LIBDIR_SUFFIX}/libscanbuild
          COMPONENT scan-build-py)
# SYCLomatic_CUSTOMIZATION begin
  list(APPEND dpct-intercept-build-dependends ${CMAKE_BINARY_DIR}/lib/libscanbuild/${lib})
  install(PROGRAMS lib/libscanbuild/${lib}
          DESTINATION lib/libscanbuild
          COMPONENT dpct-intercept-build)
# SYCLomatic_CUSTOMIZATION end
endforeach()

foreach(resource ${LibScanbuildResources})
  add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/lib/libscanbuild/resources/${resource}
                     COMMAND ${CMAKE_COMMAND} -E make_directory
                       ${CMAKE_BINARY_DIR}/lib
                     COMMAND ${CMAKE_COMMAND} -E make_directory
                       ${CMAKE_BINARY_DIR}/lib/libscanbuild
                     COMMAND ${CMAKE_COMMAND} -E make_directory
                       ${CMAKE_BINARY_DIR}/lib/libscanbuild/resources
                     COMMAND ${CMAKE_COMMAND} -E copy
                       ${CMAKE_CURRENT_SOURCE_DIR}/lib/libscanbuild/resources/${resource}
                       ${CMAKE_BINARY_DIR}/lib/libscanbuild/resources
                     DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/lib/libscanbuild/resources/${resource})
  list(APPEND Depends ${CMAKE_BINARY_DIR}/lib/libscanbuild/resources/${resource})
  install(FILES lib/libscanbuild/resources/${resource}
          DESTINATION lib${CLANG_LIBDIR_SUFFIX}/libscanbuild/resources
          COMPONENT scan-build-py)
endforeach()

foreach(lib ${LibEar})
  add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/lib/libear/${lib}
                     COMMAND ${CMAKE_COMMAND} -E make_directory
                       ${CMAKE_BINARY_DIR}/lib
                     COMMAND ${CMAKE_COMMAND} -E make_directory
                       ${CMAKE_BINARY_DIR}/lib/libear
                     COMMAND ${CMAKE_COMMAND} -E copy
                       ${CMAKE_CURRENT_SOURCE_DIR}/lib/libear/${lib}
                       ${CMAKE_BINARY_DIR}/lib/libear/
                     DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/lib/libear/${lib})
  list(APPEND Depends ${CMAKE_BINARY_DIR}/lib/libear/${lib})
  install(FILES lib/libear/${lib}
          DESTINATION lib${CLANG_LIBDIR_SUFFIX}/libear
          COMPONENT scan-build-py)
endforeach()

add_custom_target(scan-build-py ALL DEPENDS ${Depends})
add_llvm_install_targets("install-scan-build-py"
                         DEPENDS scan-build-py
                         COMPONENT scan-build-py)

if (SYCLomatic_CUSTOMIZATION)
  if (UNIX)
    add_library(ear SHARED
      ${CMAKE_CURRENT_SOURCE_DIR}/lib/libear/ear.c
    )
    target_include_directories(ear PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/lib/libear/)

    set(LIBEAR_LINKER_FLAGS "-ldl -lpthread -shared -Wl,-soname,libear.so")
    check_c_compiler_flag("-static-intel" C_SUPPORTS_STATIC_INTEL_FLAG)
    if( C_SUPPORTS_STATIC_INTEL_FLAG )
      set(LIBEAR_LINKER_FLAGS "${LIBEAR_LINKER_FLAGS} -static-intel")
    endif()

    target_link_libraries(ear PRIVATE ${LIBEAR_LINKER_FLAGS})
    target_compile_options(ear PRIVATE "-std=c99")

    install(TARGETS ear
            LIBRARY
            DESTINATION lib/libear
            COMPONENT dpct-intercept-build
    )
    list(APPEND dpct-intercept-build-dependends ear)

    add_executable(intercept-stub
      ${CMAKE_CURRENT_SOURCE_DIR}/lib/libear/intercept-stub.c
    )

    install(TARGETS intercept-stub
            DESTINATION lib/libear
            COMPONENT dpct-intercept-build
    )

    install(TARGETS intercept-stub
    DESTINATION ${CMAKE_BINARY_DIR}/lib/libear
    COMPONENT dpct-intercept-build
    )

    list(APPEND dpct-intercept-build-dependends intercept-stub)
    list(APPEND dpct-intercept-build-dependends lib/libear/intercept-stub)

    list(APPEND dpct-intercept-build-dependends ${CMAKE_BINARY_DIR}/lib/libear/__init__.py)
    install(PROGRAMS lib/libear/__init__.py
            DESTINATION lib/libear
            COMPONENT dpct-intercept-build)
  endif()

  install(PROGRAMS bin/intercept-build
                   DESTINATION bin
                   COMPONENT dpct-intercept-build)

  list(APPEND dpct-intercept-build-dependends ${CMAKE_BINARY_DIR}/bin/intercept-build)

  add_custom_target(dpct-intercept-build ALL DEPENDS ${dpct-intercept-build-dependends})

  if (NOT CMAKE_CONFIGURATION_TYPES) # don't add this for IDE's.
    add_llvm_install_targets(install-dpct-intercept-build
                             DEPENDS dpct-intercept-build
                             COMPONENT dpct-intercept-build)
  endif()
endif()
