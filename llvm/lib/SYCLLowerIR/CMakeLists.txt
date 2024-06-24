# INTEL_CUSTOMIZATION
#
# INTEL CONFIDENTIAL
#
# Modifications, Copyright (C) 2021 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials, and
# your use of them is governed by the express license under which they were
# provided to you ("License"). Unless the License provides otherwise, you may not
# use, modify, copy, publish, distribute, disclose or transmit this software or
# the related documents without Intel's prior written permission.
#
# This software and the related documents are provided as is, with no express
# or implied warranties, other than those that are expressly stated in the
# License.
#
# end INTEL_CUSTOMIZATION

# Lowering of SYCL ESIMD kernels depends on vc-intrinsics
# NOTE: could have been added earlier from llvm/projects
if (NOT TARGET LLVMGenXIntrinsics)
  if (NOT DEFINED LLVMGenXIntrinsics_SOURCE_DIR)
    set(LLVMGenXIntrinsics_GIT_REPO https://github.com/intel/vc-intrinsics.git)

    # Date: 08 May 2024
    # Fix warning from switch
    set(LLVMGenXIntrinsics_GIT_TAG b2565a03eb3cac07f5e8000fde971f95dc782c75)

    if (INTEL_CUSTOMIZATION)
      # for xmain builds the LLVMGenXIntrinsics_GIT_REPO is constructed from
      # GIT_REPO_PATH and GIT_REPO_TAG set independently depending on build
      # configuration (public/embargo) and condition (building from under ICS
      # or not)

      # default repo path for public xmain builds, not using ICS
      set(LLVMGenXIntrinsics_GIT_REPO_PATH https://github.com/intel)
      set(LLVMGenXIntrinsics_GIT_REPO_NAME vc-intrinsics.git)

      # Date:   Dec 14, 2023
      # Add support for XeHPCVG platform
      set(LLVMGenXIntrinsics_GIT_TAG da892e1982b6c25b9a133f85b4ac97142d8a3def)

      # Determine if the build is embargo and provide specific settings.
      set(IS_INTERNAL_ESIMD_BUILD FALSE)

      if (DEFINED ENV{ICS_GIT_MIRROR} AND NOT "$ENV{ICS_GIT_MIRROR}" STREQUAL "")
        # ICS_GIT_MIRROR, if set, overrides GIT_REPO_PATH.
        set(LLVMGenXIntrinsics_GIT_REPO_PATH $ENV{ICS_GIT_MIRROR})
        STRING(REGEX REPLACE "\\\\" "/" LLVMGenXIntrinsics_GIT_REPO_PATH "${LLVMGenXIntrinsics_GIT_REPO_PATH}")
# INTEL_FEATURE_ESIMD_EMBARGO
      elseif (IS_INTERNAL_ESIMD_BUILD)
        # ICS_GIT_MIRROR is obligatory for embargo builds.
        message( FATAL_ERROR "xmain ESIMD_EMBARGO build requires ICS_GIT_MIRROR env var to be set" )
# end INTEL_FEATURE_ESIMD_EMBARGO
      endif ()

      # Construct GIT_REPO from from GIT_REPO_PATH and GIT_REPO_NAME.
      set( LLVMGenXIntrinsics_GIT_REPO
        ${LLVMGenXIntrinsics_GIT_REPO_PATH}/${LLVMGenXIntrinsics_GIT_REPO_NAME} )
    endif (INTEL_CUSTOMIZATION)

    message(STATUS "vc-intrinsics repo is missing. Will try to download it from ${LLVMGenXIntrinsics_GIT_REPO}")
    include(FetchContent)
    FetchContent_Declare(vc-intrinsics
      GIT_REPOSITORY ${LLVMGenXIntrinsics_GIT_REPO}
      GIT_TAG        ${LLVMGenXIntrinsics_GIT_TAG}
    )
    FetchContent_MakeAvailable(vc-intrinsics)
    FetchContent_GetProperties(vc-intrinsics)

    set(LLVMGenXIntrinsics_SOURCE_DIR ${vc-intrinsics_SOURCE_DIR})
    set(LLVMGenXIntrinsics_BINARY_DIR ${vc-intrinsics_BINARY_DIR})
  else (NOT DEFINED LLVMGenXIntrinsics_SOURCE_DIR)
    # -DLLVMGenXIntrinsics_SOURCE_DIR is provided
    message(STATUS "vc-intrinsics are added manually ${LLVMGenXIntrinsics_SOURCE_DIR}")

    set(LLVMGenXIntrinsics_BINARY_DIR ${CMAKE_BINARY_DIR}/vc-intrinsics-build)
    add_subdirectory(${LLVMGenXIntrinsics_SOURCE_DIR} ${LLVMGenXIntrinsics_BINARY_DIR})
  endif (NOT DEFINED LLVMGenXIntrinsics_SOURCE_DIR)

  target_include_directories(LLVMGenXIntrinsics
    PUBLIC $<BUILD_INTERFACE:${LLVMGenXIntrinsics_SOURCE_DIR}/GenXIntrinsics/include>
    PUBLIC $<BUILD_INTERFACE:${LLVMGenXIntrinsics_BINARY_DIR}/GenXIntrinsics/include>
  )
endif (NOT TARGET LLVMGenXIntrinsics)

set_property(GLOBAL PROPERTY LLVMGenXIntrinsics_SOURCE_PROP ${LLVMGenXIntrinsics_SOURCE_DIR})
set_property(GLOBAL PROPERTY LLVMGenXIntrinsics_BINARY_PROP ${LLVMGenXIntrinsics_BINARY_DIR})

# INTEL_CUSTOMIZATION
# Disable deprecated-declarations warnings because they are coming from
# external GenXIntrinsics headers.
if (NOT WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
else(NOT WIN32)
  add_definitions(-wd4996)
endif (NOT WIN32)
#end INTEL_CUSTOMIZATION

add_llvm_component_library(LLVMSYCLLowerIR
  ESIMD/ESIMDOptimizeVecArgCallConv.cpp
  ESIMD/ESIMDUtils.cpp
  ESIMD/ESIMDVerifier.cpp
  ESIMD/ESIMDRemoveHostCode.cpp
  ESIMD/ESIMDRemoveOptnoneNoinline.cpp
  ESIMD/LowerESIMD.cpp
  ESIMD/LowerESIMDKernelAttrs.cpp
  RecordSYCLAspectNames.cpp
  CleanupSYCLMetadata.cpp
  CompileTimePropertiesPass.cpp
  DeviceGlobals.cpp
  ESIMD/LowerESIMDVLoadVStore.cpp
  ESIMD/LowerESIMDSlmReservation.cpp
  HostPipes.cpp
  LowerInvokeSimd.cpp
  LowerWGLocalMemory.cpp
  LowerWGScope.cpp
  ModuleSplitter.cpp
  MutatePrintfAddrspace.cpp
  SYCLAddOptLevelAttribute.cpp
  SYCLDeviceRequirements.cpp
  SYCLPropagateAspectsUsage.cpp
  SYCLPropagateJointMatrixUsage.cpp
  SYCLUtils.cpp
  SanitizeDeviceGlobal.cpp

  LocalAccessorToSharedMemory.cpp
  GlobalOffset.cpp
  TargetHelpers.cpp
  PrepareSYCLNativeCPU.cpp
  RenameKernelSYCLNativeCPU.cpp
  ConvertToMuxBuiltinsSYCLNativeCPU.cpp
  PipelineSYCLNativeCPU.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/SYCLLowerIR
  ${LLVM_MAIN_SRC_DIR}/projects/vc-intrinsics/GenXIntrinsics/include
  ${LLVM_BINARY_DIR}/projects/vc-intrinsics/GenXIntrinsics/include

  DEPENDS
  intrinsics_gen
  LLVMGenXIntrinsics
  LLVMDemangle
  LLVMTransformUtils
  DeviceConfigFile

  LINK_LIBS
  LLVMGenXIntrinsics
  LLVMDemangle
  LLVMTargetParser
  LLVMTransformUtils

  LINK_COMPONENTS
  Analysis
  BitWriter
  Core
  IRPrinter
  Support
  ipo
  )

target_include_directories(LLVMSYCLLowerIR
  PRIVATE ${LLVM_MAIN_SRC_DIR}/projects/vc-intrinsics/GenXIntrinsics/include
  PRIVATE ${LLVM_BINARY_DIR}/projects/vc-intrinsics/GenXIntrinsics/include)

## Remove the Werror for external component GenXIntrinsics.
if(MSVC)
  target_compile_options(LLVMGenXIntrinsics  PRIVATE "-WX-")
else()
  target_compile_options(LLVMGenXIntrinsics  PRIVATE "-Wno-error")
endif()
