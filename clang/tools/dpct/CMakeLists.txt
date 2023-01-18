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
  DPCT.cpp
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
