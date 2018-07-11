set(BinFiles
    intercept-build)

set(LibScanBuildFiles
    arguments.py
    clang.py
    compilation.py
    __init__.py
    intercept.py
    shell.py
)

set(LibearFiles
    config.h.in
    ear.c
    __init__.py
)


foreach(BinFile ${BinFiles})
  add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/bin/${BinFile}
                     COMMAND ${CMAKE_COMMAND} -E make_directory
                       ${CMAKE_BINARY_DIR}/bin
                     COMMAND ${CMAKE_COMMAND} -E copy
                       ${CMAKE_CURRENT_SOURCE_DIR}/bin/${BinFile}
                       ${CMAKE_BINARY_DIR}/bin/
                     DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/bin/${BinFile})
  list(APPEND Depends ${CMAKE_BINARY_DIR}/bin/${BinFile})
  install(PROGRAMS bin/${BinFile} DESTINATION bin)
endforeach()

foreach(LibScanBuildFile ${LibScanBuildFiles})
  add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/libscanbuild/${LibScanBuildFile}
                     COMMAND ${CMAKE_COMMAND} -E make_directory
                       ${CMAKE_BINARY_DIR}/libscanbuild
                     COMMAND ${CMAKE_COMMAND} -E copy
                       ${CMAKE_CURRENT_SOURCE_DIR}/libscanbuild/${LibScanBuildFile}
                       ${CMAKE_BINARY_DIR}/libscanbuild/
                     DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/libscanbuild/${LibScanBuildFile})
  list(APPEND Depends ${CMAKE_BINARY_DIR}/libscanbuild/${LibScanBuildFile})
  install(PROGRAMS libscanbuild/${LibScanBuildFile} DESTINATION libscanbuild)
endforeach()

foreach(LibearFile ${LibearFiles})
  add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/libear/${LibearFile}
                     COMMAND ${CMAKE_COMMAND} -E make_directory
                       ${CMAKE_BINARY_DIR}/libear
                     COMMAND ${CMAKE_COMMAND} -E copy
                       ${CMAKE_CURRENT_SOURCE_DIR}/libear/${LibearFile}
                       ${CMAKE_BINARY_DIR}/libear/
                     DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/libear/${LibearFile})
  list(APPEND Depends ${CMAKE_BINARY_DIR}/libear/${LibearFile})
  install(PROGRAMS libear/${LibearFile} DESTINATION libear)
endforeach()

add_custom_target(scan-build-py DEPENDS ${Depends})
set_target_properties(scan-build-py PROPERTIES FOLDER "Misc")
