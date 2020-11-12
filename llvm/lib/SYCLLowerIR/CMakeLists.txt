# Lowering of SYCL ESIMD kernels depends on vc-intrinsics
# NOTE: could have been added earlier from llvm/projects
if (NOT TARGET LLVMGenXIntrinsics)
  if (NOT DEFINED LLVMGenXIntrinsics_SOURCE_DIR)
    set( LLVMGenXIntrinsics_GIT_REPO
         https://github.com/intel/vc-intrinsics.git )
    set( LLVMGENXIntrinsics_GIT_TAG 91f3df2cd87a7b9706df32046c34fe8d07ecf854)
    if( INTEL_CUSTOMIZATION )
      if( DEFINED ENV{ICS_GIT_MIRROR} AND NOT "$ENV{ICS_GIT_MIRROR}" STREQUAL "" )
        set( LLVMGenXIntrinsics_GIT_REPO
             $ENV{ICS_GIT_MIRROR}/dpd_icl-vc-intrinsics.git )
        STRING(REGEX REPLACE "\\\\" "/" LLVMGenXIntrinsics_GIT_REPO "${LLVMGenXIntrinsics_GIT_REPO}")
        set( LLVMGENXIntrinsics_GIT_TAG 4143fa153a471c4d54969a10bca472a53088f556)
      endif()
    endif(INTEL_CUSTOMIZATION)

    message(STATUS "vc-intrinsics are missing. Will try to download them from ${LLVMGenXIntrinsics_GIT_REPO} ") #INTEL_CUSTOMIZATION
    include(FetchContent)
    FetchContent_Declare(vc-intrinsics
      GIT_REPOSITORY ${LLVMGenXIntrinsics_GIT_REPO} #INTEL_CUSTOMIZATION
      GIT_TAG        ${LLVMGENXIntrinsics_GIT_TAG}  #INTEL_CUSTOMIZATION
    )
    FetchContent_MakeAvailable(vc-intrinsics)
    FetchContent_GetProperties(vc-intrinsics)

    set(LLVMGenXIntrinsics_SOURCE_DIR ${vc-intrinsics_SOURCE_DIR})
    set(LLVMGenXIntrinsics_BINARY_DIR ${vc-intrinsics_BINARY_DIR})
  else()
    # -DLLVMGenXIntrinsics_SOURCE_DIR is provided
    message(STATUS "vc-intrinsics are added manually ${LLVMGenXIntrinsics_SOURCE_DIR}")

    set(LLVMGenXIntrinsics_BINARY_DIR ${CMAKE_BINARY_DIR}/vc-intrinsics-build)
    add_subdirectory(${LLVMGenXIntrinsics_SOURCE_DIR} ${LLVMGenXIntrinsics_BINARY_DIR})
  endif()

  target_include_directories(LLVMGenXIntrinsics
    PUBLIC $<BUILD_INTERFACE:${LLVMGenXIntrinsics_SOURCE_DIR}/GenXIntrinsics/include>
    PUBLIC $<BUILD_INTERFACE:${LLVMGenXIntrinsics_BINARY_DIR}/GenXIntrinsics/include>
  )
endif()

set_property(GLOBAL PROPERTY LLVMGenXIntrinsics_SOURCE_PROP ${LLVMGenXIntrinsics_SOURCE_DIR})
set_property(GLOBAL PROPERTY LLVMGenXIntrinsics_BINARY_PROP ${LLVMGenXIntrinsics_BINARY_DIR})

add_llvm_component_library(LLVMSYCLLowerIR
  LowerWGScope.cpp
  LowerESIMD.cpp
  LowerESIMDVLoadVStore.cpp
  LowerESIMDVecArg.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/SYCLLowerIR
  ${LLVM_MAIN_SRC_DIR}/projects/vc-intrinsics/GenXIntrinsics/include
  ${LLVM_BINARY_DIR}/projects/vc-intrinsics/GenXIntrinsics/include

  DEPENDS
  intrinsics_gen
  LLVMGenXIntrinsics
  LLVMDemangle
  LLVMTransformUtils

  LINK_LIBS
  LLVMGenXIntrinsics
  LLVMDemangle
  LLVMTransformUtils
  )

target_include_directories(LLVMSYCLLowerIR
  PRIVATE ${LLVM_MAIN_SRC_DIR}/projects/vc-intrinsics/GenXIntrinsics/include
  PRIVATE ${LLVM_BINARY_DIR}/projects/vc-intrinsics/GenXIntrinsics/include)
