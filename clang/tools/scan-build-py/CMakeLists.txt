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

#INTEL_CUSTOMIZATION begin
if(UNIX)
set(LibearFiles
    __init__.py
)
set(EARTARGET ear)
add_library(ear SHARED
    libear/ear.c
)
target_include_directories(ear PRIVATE libear/)
target_link_libraries(ear PRIVATE
    "-ldl -lpthread -shared -Wl,-soname,libear.so")
target_compile_options(ear PRIVATE
    "-DNDEBUG -O3 -fPIC -std=c99 -D_GNU_SOURCE")
install(TARGETS ear
    LIBRARY
    DESTINATION libear
    COMPONENT dpct-libear-files
)

set(INTERCEPT_STUB_TARGET intercept-stub)
add_executable(intercept-stub
  libear/intercept-stub.c
  )

else()
set(EARTARGET "")
#windows, default install the binary
set(LibearFiles
   config.h.in
   ear.c
   __init__.py
)
endif()
#INTEL_CUSTOMIZATION end

set(OutPutBinFiles)
foreach(BinFile ${BinFiles})
  add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/bin/${BinFile}
                     COMMAND ${CMAKE_COMMAND} -E make_directory
                       ${CMAKE_BINARY_DIR}/bin
                     COMMAND ${CMAKE_COMMAND} -E copy
                       ${CMAKE_CURRENT_SOURCE_DIR}/bin/${BinFile}
                       ${CMAKE_BINARY_DIR}/bin/
                     DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/bin/${BinFile})
  list(APPEND Depends ${CMAKE_BINARY_DIR}/bin/${BinFile})
  list(APPEND OutPutBinFiles ${CMAKE_CURRENT_SOURCE_DIR}/bin/${BinFile})
  install(PROGRAMS bin/${BinFile} DESTINATION bin)
endforeach()

set(OutPutLibScanBuildFiles)
foreach(LibScanBuildFile ${LibScanBuildFiles})
  add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/libscanbuild/${LibScanBuildFile}
                     COMMAND ${CMAKE_COMMAND} -E make_directory
                       ${CMAKE_BINARY_DIR}/libscanbuild
                     COMMAND ${CMAKE_COMMAND} -E copy
                       ${CMAKE_CURRENT_SOURCE_DIR}/libscanbuild/${LibScanBuildFile}
                       ${CMAKE_BINARY_DIR}/libscanbuild/
                     DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/libscanbuild/${LibScanBuildFile})
  list(APPEND Depends ${CMAKE_BINARY_DIR}/libscanbuild/${LibScanBuildFile})
  list(APPEND OutPutLibScanBuildFiles ${CMAKE_BINARY_DIR}/libscanbuild/${LibScanBuildFile})
  install(PROGRAMS libscanbuild/${LibScanBuildFile} DESTINATION libscanbuild)
endforeach()

set(OutPutLibearFile)
foreach(LibearFile ${LibearFiles})
  add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/libear/${LibearFile}
                     COMMAND ${CMAKE_COMMAND} -E make_directory
                       ${CMAKE_BINARY_DIR}/libear
                     COMMAND ${CMAKE_COMMAND} -E copy
                       ${CMAKE_CURRENT_SOURCE_DIR}/libear/${LibearFile}
                       ${CMAKE_BINARY_DIR}/libear/
                     DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/libear/${LibearFile})
  list(APPEND Depends ${CMAKE_BINARY_DIR}/libear/${LibearFile})
  list(APPEND OutPutLibearFile ${CMAKE_BINARY_DIR}/libear/${LibearFile})
if(UNIX)
  list(APPEND OutPutLibearFile ${CMAKE_BINARY_DIR}/bin/intercept-stub)
endif()
  install(PROGRAMS libear/${LibearFile} DESTINATION libear)
endforeach()

add_custom_target(scan-build-py DEPENDS ${Depends})
set_target_properties(scan-build-py PROPERTIES FOLDER "Misc")

install(
  PROGRAMS ${OutPutBinFiles}
  COMPONENT dpct-intercept-build
  DESTINATION bin)
add_custom_target(dpct-intercept-build ALL DEPENDS ${OutPutLibScanBuildFiles})
if (NOT CMAKE_CONFIGURATION_TYPES) # don't add this for IDE's.
  add_llvm_install_targets(install-dpct-intercept-build
                           DEPENDS dpct-intercept-build
                           COMPONENT dpct-intercept-build)
endif()


install(
  PROGRAMS ${OutPutLibScanBuildFiles}
  COMPONENT dpct-lib-scan-build-files
  DESTINATION libscanbuild)

add_custom_target(dpct-lib-scan-build-files ALL DEPENDS ${OutPutLibScanBuildFiles})
if (NOT CMAKE_CONFIGURATION_TYPES) # don't add this for IDE's.
  add_llvm_install_targets(install-dpct-lib-scan-build-files
                           DEPENDS dpct-lib-scan-build-files
                           COMPONENT dpct-lib-scan-build-files)
endif()

install(
  PROGRAMS ${OutPutLibearFile}
  COMPONENT dpct-libear-files
  DESTINATION libear)

add_custom_target(dpct-libear-files ALL DEPENDS ${OutPutLibearFile} ${EARTARGET} ${INTERCEPT_STUB_TARGET})
if (NOT CMAKE_CONFIGURATION_TYPES) # don't add this for IDE's.
  add_llvm_install_targets(install-dpct-libear-files
                           DEPENDS dpct-libear-files ${EARTARGET}
                           COMPONENT dpct-libear-files)
endif()
