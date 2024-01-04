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
else()
    set(DNN_LIB "dnnl")
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
<<<<<<< HEAD
=======

# Checks if the arg is an option
macro(CHECK_OPTION _arg _is_opt)
  if(${_arg} STREQUAL "REQUIRED" OR
    ${_arg} STREQUAL "QUIET" OR
    ${_arg} STREQUAL "EXACT")
    set(_is_opt TRUE)
  endif()
endmacro()

# Find oneAPI lib packages
macro(DPCT_HELPER_FIND_PACKAGE package)
  set(_sycl_targets "")
  set(_cuda_package_version "")
  set(_sycl_package_version "")
  set(_cuda_package_option "")
  set(_sycl_package_option "")

  set(_pkg_args ${ARGN})
  set(_pkg_args_cnt 0)
  
  # Parse CUDA package version
  list(LENGTH _pkg_args _pkg_args_cnt)
  if(_pkg_args_cnt GREATER 0)
    set(_is_ver FALSE)
    list(GET _pkg_args 0 _arg0)
    
    check_version(${_arg0} _is_ver)

    if(${_is_ver})
      list(POP_FRONT _pkg_args _cuda_package_version)
      list(LENGTH _pkg_args _pkg_args_cnt)
    endif()
  endif()

  # Parse CUDA package option
  if(_pkg_args_cnt GREATER 0)
    set(_is_opt FALSE)
    list(GET _pkg_args 0 _arg0)

    check_option(${_arg0} _is_opt)

    if(${_is_opt})
      list(POP_FRONT _pkg_args _cuda_package_option)
      list(LENGTH _pkg_args _pkg_args_cnt)
    endif()

    if(NOT _cuda_package_option STREQUAL "EXACT")
      set(_sycl_package_option ${_cuda_package_option})
    endif()
  endif()
  
  # Identify SYCL equivalents for CUDA packages
  if(${package} STREQUAL "IntelSYCL")
    list(APPEND _sycl_targets "IntelSYCL")
  elseif(${package} STREQUAL "oneAPIToolkit")
    DPCT_GET_SOURCES(_cuda_imp_targets ${_pkg_args})

    if(_pkg_args_cnt GREATER 0)
      foreach(_cuda_target _cuda_imp_targets)
        if(${_cuda_target} STREQUAL "blas"
          ${_cuda_target} STREQUAL "rng"
          ${_cuda_target} STREQUAL "sparse"
          ${_cuda_target} STREQUAL "solver"
          ${_cuda_target} STREQUAL "fft")
          list(APPEND _sycl_targets "MKL")
        endif()
      endforeach()
    else()
      list(APPEND _sycl_targets "MKL")
    endif()
  elseif(${package} STREQUAL "oneDPL")
    list(APPEND _sycl_targets "oneDPL")
  endif()

  foreach(_sycl_package ${_sycl_targets})
    find_package(${_sycl_package} ${_sycl_package_version} ${${_sycl_package_option}})
  endforeach()

endmacro()
>>>>>>> 1ece66ee9140 (Updated dpct helper macro logic)
