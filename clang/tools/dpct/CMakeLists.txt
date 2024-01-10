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

if(TRUE)
  message("yes yes yes")
  message("yes yes yes")
  message("yes yes yes")
  add_clang_tool(dpct-binary-tmp
    DPCT.cpp NO_INSTALL_RPATH
    )
  set_target_properties(dpct-binary-tmp PROPERTIES OUTPUT_NAME "dpct-tmp")
  target_link_libraries(dpct-binary-tmp
    PRIVATE
    DPCT_TMP
    )
  add_custom_target(gen_mapping_str
    COMMAND "${CMAKE_COMMAND}"
      -DTARGET="${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/APIMappingRegisterFast.def"
      -DSRC_DIRECTORY="${CMAKE_CURRENT_SOURCE_DIR}/../../examples/DPCT/"
      -P ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/DPCT/APIMapping/GenerateRegisterFast.cmake
    DEPENDS dpct-binary-tmp)
  add_clang_tool(dpct-binary
    DPCT.cpp NO_INSTALL_RPATH
    DEPENDS gen_mapping_str
    )
else()
  message("not not not")
  message("not not not")
  message("not not not")
  add_clang_tool(dpct-binary
    DPCT.cpp NO_INSTALL_RPATH
    )
endif()

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

if(UNIX)
  set(dpct_vars_script vars.sh_SYCLomatic)
  set(dpct_setvars_script "setvars.sh")
else()
  set(dpct_vars_script vars.bat_SYCLomatic)
  set(dpct_setvars_script "setvars.bat")
endif()

install(
  FILES ${dpct_vars_script}
  COMPONENT dpct-setvars
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  DESTINATION ./
  RENAME ${dpct_setvars_script})

if (NOT CMAKE_CONFIGURATION_TYPES)
    add_llvm_install_targets(install-dpct-setvars
                             COMPONENT dpct-setvars)
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

# set(API_NAME_SET)
# foreach(FILE ${SRC_FILES})
#   file(READ ${FILE} FILE_STR)
#   string(REGEX REPLACE ".*/(.*).cu$" "\\1" API_NAME ${FILE})
#   list(FIND API_NAME_SET ${API_NAME} INDEX)
#   if(${INDEX} GREATER_EQUAL 0)
#     message(FATAL_ERROR "The API Mapping cases name " ${API_NAME} " is duplicated.")
#   endif()
#   list(APPEND API_NAME_SET ${API_NAME})
#   file(APPEND ${API_MAPPING_REGISTER_FOLDER}APIMappingRegister.def
#   "registerEntry(\"" "${API_NAME}" "\",\nR\"(" "${FILE_STR}" ")\");\n")
# endforeach()

# install(
#   FILES ${dpct_cmake_helper_file}
#   COMPONENT dpct-cmake-file
#   PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
#   DESTINATION ./cmake)
  
if (NOT CMAKE_CONFIGURATION_TYPES) # don't add this for IDE's.
  add_llvm_install_targets(install-dpct-cmake-file
                           COMPONENT dpct-cmake-file)
endif()
