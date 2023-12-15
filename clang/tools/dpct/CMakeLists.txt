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

add_subdirectory(DpctOptRules)

set(dpct_cmake_helper_file
  ${CMAKE_SOURCE_DIR}/../clang/tools/dpct/cmake/dpct.cmake
)

install(
  FILES ${dpct_cmake_helper_file}
  COMPONENT dpct-cmake-file
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
  DESTINATION ./cmake)

if (NOT CMAKE_CONFIGURATION_TYPES) # don't add this for IDE's.
  add_llvm_install_targets(install-dpct-cmake-file
                           COMPONENT dpct-cmake-file)
endif()
