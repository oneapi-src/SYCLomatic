#==---- dpct.cmake --------------------------------- cmake script file ----==//
#
# Copyright (C) Intel Corporation
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# See https://llvm.org/LICENSE.txt for license information.
#
#===----------------------------------------------------------------------===//

macro(_DPCT_GET_SOURCES _sources)
  set( ${_sources} )
  foreach(arg ${ARGN})
    # Assume arg is a source file
    list(APPEND ${_sources} ${arg})
  endforeach()
endmacro()

macro(_DPCT_CREATE_BUILD_COMMAND sycl_target generated_files)
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

macro(_DPCT_COMPILE_SYCL_CODE_IMP sycl_target generated_files)
  set(_sycl_target "${sycl_target}")
  _DPCT_GET_SOURCES(_sources ${ARGN})

  # Create custom command for each cpp source file
  _DPCT_CREATE_BUILD_COMMAND( ${_sycl_target} _generated_files ${_sources})

  set( ${generated_files} ${_generated_files})
endmacro()

# Return generated device code files from input SYCL source files
macro(DPCT_HELPER_COMPILE_SYCL_CODE generated_files)
  _DPCT_COMPILE_SYCL_CODE_IMP(sycl_device ${generated_files} ${ARGN})
endmacro()

# Always set SYCL_HAS_FP16 to true to assume SYCL device to support float16
message("dpct.cmake: SYCL_HAS_FP16 is set true by default.")
set(SYCL_HAS_FP16 TRUE)

# Return the list of object file paths generated for the given SYCL source files
macro(DPCT_HELPER_SYCL_COMPILE generated_files)
  _DPCT_GET_SOURCES(_sources ${ARGN})

  # can't continue without list of source files
  if("${_sources}" STREQUAL "")
    message(FATAL "Failed to find the source files while running the macro 'DPCT_HELPER_SYCL_COMPILE'")
  endif()

  _DPCT_CREATE_BUILD_COMMAND("sycl_device" ${generated_files} ${_sources})
endmacro()

if(WIN32)
    set(DNN_LIB "dnnl.lib")
    set(MKL_LIB "-Qmkl")
else()
    set(DNN_LIB "dnnl")
    set(MKL_LIB "-qmkl")
endif()

# Link MKL library to target
macro(DPCT_HELPER_ADD_MKL_TO_TARGET target)
  if(WIN32)
    target_compile_options(${target} PUBLIC -fsycl /DMKL_ILP64 /Qmkl:parallel /Qtbb /MD)
    target_link_libraries(${target} PUBLIC -fsycl OpenCL.lib)
  elseif(UNIX AND NOT APPLE)
    target_compile_options(${target} PUBLIC -fsycl -DMKL_ILP64 -qmkl=parallel -qtbb)
    target_link_libraries(${target} PUBLIC -qmkl=parallel -qtbb -fsycl)
  else()
    message(FATAL_ERROR "Unsupported platform")
  endif()
endmacro()

set(CMAKE_SYCL_COMPILER "icpx")
set(CMAKE_SYCL_ARCHITECTURES "")
set(CMAKE_SYCL_FLAGS "")
set(COMPATIBILITY_VERSION 99.9)
set(COMPATIBILITY_VERSION_MAJOR 99)
set(COMPATIBILITY_VERSION_MINOR 9)
set(COMPATIBILITY_VERSION_STRING "${COMPATIBILITY_VERSION}")
set(SYCL_TOOLKIT_ROOT_DIR "${ONEAPI_ROOT}")
set(SYCL_TOOLKIT_INCLUDE "${SYCL_INCLUDE_DIR}")
# The SYCL runtime library is auto-loaded when option `-fsycl` is specified. So we can
# safely set the CUDAToolkit_LIBRARY_DIR to ""
set(SYCLToolkit_LIBRARY_DIR "${SYCL_INCLUDE_DIR}/../lib")
set(SYCL_HOST_COMPILER "icpx")
set(SYCL_HOST_FLAGS "")

# 'SYCL_COMPILER_EXECUTABLE' is used to specify the path to the SYCL Compiler (icpx).
set(SYCL_COMPILER_EXECUTABLE "${SYCL_INCLUDE_DIR}/../bin/icpx")
set(SYCL_PROPAGATE_HOST_FLAGS "")
