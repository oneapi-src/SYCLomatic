#==---- dpct.cmake --------------------------------- cmake script file ----==//
#
# Copyright (C) Intel Corporation
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# See https://llvm.org/LICENSE.txt for license information.
#
#===----------------------------------------------------------------------===//

macro(DPCT_GET_SOURCES _sources)
  set( ${_sources} )
  foreach(arg ${ARGN})
    # Assume arg is a source file
    list(APPEND ${_sources} ${arg})
  endforeach()
endmacro()

macro(DPCT_CREATE_BUILD_COMMAND sycl_target generated_files)
  set(_argn_list "${ARGN}")
  set(_generated_files "")
  set(generated_extension ${CMAKE_CXX_OUTPUT_EXTENSION})

  set(_counter 1) # use to unique the generated obj file name
  set(_sycl_target "${sycl_target}")
  set(options)
  if (UNIX)
    set(options -fPIC -shared)
  else()
    set(options -shared)
  endif()

  # Iterate each macro arguments and create custom command for each cpp file
  foreach(file ${_argn_list})
    if(${file} MATCHES "\\.cpp")
        get_filename_component( basename ${file} NAME )
    
        set(generated_file_basename "${sycl_target}_${_counter}_generated_${basename}${generated_extension}")
        set(generated_file "${CMAKE_CURRENT_BINARY_DIR}/${generated_file_basename}")

        get_filename_component(file_path "${file}" PATH)
        if(IS_ABSOLUTE "${file_path}")
            set(source_file "${file}")
        else()
            set(source_file "${CMAKE_CURRENT_SOURCE_DIR}/${file}")
        endif()

        add_custom_command(
          OUTPUT ${generated_file}
          COMMAND icpx -fsycl  ${options} -o ${generated_file} ${source_file}
          DEPENDS ${file}
        )

        list(APPEND _generated_files ${generated_file})
    else()
        message(FATAL_ERROR "Only support cpp file.")
    endif()

    math(EXPR _counter "${_counter} + 1")
  endforeach()

  set(${generated_files} ${_generated_files})
endmacro()

macro(DPCT_COMPILE_SYCL_CODE_IMP sycl_target generated_files)
  set(_sycl_target "${sycl_target}")
  DPCT_GET_SOURCES(_sources ${ARGN})

  # Create custom command for each cpp source file
  DPCT_CREATE_BUILD_COMMAND( ${_sycl_target} _generated_files ${_sources})

  set( ${generated_files} ${_generated_files})
endmacro()

# Return generated device code files from input SYCL source files
macro(DPCT_COMPILE_SYCL_CODE generated_files)
  DPCT_COMPILE_SYCL_CODE_IMP(sycl_device ${generated_files} ${ARGN})
endmacro()

# Always set SYCL_HAS_FP16 to true to assume SYCL device to support float16
message("dpct.cmake: SYCL_HAS_FP16 is set true by default.")
set(SYCL_HAS_FP16 TRUE)

# Replaces cuda_compile with it's equivalent SYCL repr
macro(DPCT_COMPILE generated_files)
  # list of dp.cpp source files
  set(_sources "")
  # type of library specified in dpct_compile
  set(_lib_type "")
  set(_possible_lib_types "STATIC;SHARED;MODULE")

  # segregate source files, library type and ignore nvcc options
  foreach(arg ${ARGN})
    # to confirm: what about cpp files?
    if(${arg} MATCHES "\\.dp\\.(cpp|hpp)$")
      list(APPEND _sources "${arg}")

    elseif(${arg} IN_LIST _possible_lib_types)
      set(_lib_type "${arg}")

    elseif(${arg} STREQUAL "OPTIONS")
      # we don't need to pass nvcc options to icpx
      # to confirm: are there nvcc options compatible with icpx?
      break()

    else()
      message(VERBOSE "Ignoring '${arg}' passed to dpct_compile")

    endif()
  endforeach()

  # can't continue without list of source files
  if("${_sources}" STREQUAL "")
    message(FATAL "Failed to find source files while migrating `cuda_compile`")
  endif()

  # if library type is not specified then the build type is either SHARED or
  # STATIC based on the cmake variable BUILD_SHARED_LIBS
  if ("${_lib_type}" STREQUAL "")
    add_library(${generated_files} ${_sources})

  else()
    add_library(${generated_files} ${_lib_type} ${_sources})

  endif()
endmacro()

