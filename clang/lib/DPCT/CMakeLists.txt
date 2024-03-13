set(LLVM_LINK_COMPONENTS
  Option
  Support
  )

set(RUNTIME_HEADERS
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpct/atomic.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpct/bindless_images.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpct/device.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpct/image.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpct/math.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpct/memory.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpct/util.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpct/blas_utils.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpct/dnnl_utils.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpct/dpct.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpct/dpl_utils.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpct/kernel.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpct/rng_utils.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpct/lib_common_utils.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpct/ccl_utils.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpct/sparse_utils.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpct/fft_utils.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpct/lapack_utils.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpct/dpl_extras/algorithm.h
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpct/dpl_extras/functional.h
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpct/dpl_extras/iterators.h
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpct/dpl_extras/memory.h
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpct/dpl_extras/numeric.h
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpct/dpl_extras/vector.h
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpct/dpl_extras/dpcpp_extensions.h
  )

set(PROCESS_FILES_OUTPUT
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dpl_extras/algorithm.h.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dpl_extras/dpcpp_extensions.h.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dpl_extras/functional.h.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dpl_extras/iterators.h.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dpl_extras/memory.h.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dpl_extras/numeric.h.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dpl_extras/vector.h.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/atomic.hpp.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/bindless_images.hpp.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/blas_utils.hpp.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dnnl_utils.hpp.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/device.hpp.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dpct.hpp.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dpl_utils.hpp.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/image.hpp.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/kernel.hpp.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/math.hpp.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/memory.hpp.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/util.hpp.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/rng_utils.hpp.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/lib_common_utils.hpp.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/ccl_utils.hpp.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/sparse_utils.hpp.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/fft_utils.hpp.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/lapack_utils.hpp.inc
)

add_custom_command(
  OUTPUT Confusables.inc
  COMMAND make_confusable_table ${CMAKE_CURRENT_SOURCE_DIR}/ConfusableTable/confusables.txt ${CMAKE_CURRENT_BINARY_DIR}/Confusables.inc
  DEPENDS make_confusable_table ConfusableTable/confusables.txt
  )

add_custom_command(
  DEPENDS ${RUNTIME_HEADERS} ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/processFiles.py
  OUTPUT  ${PROCESS_FILES_OUTPUT}
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/processFiles.py --build-dir ${CMAKE_BINARY_DIR}
  )

add_custom_target(genconfusable 
  DEPENDS Confusables.inc
  )

add_custom_target(dpct_helper_headers_and_inc
  DEPENDS ${PROCESS_FILES_OUTPUT}
  )

# Handle API mapping cases header
set(API_MAPPING_REGISTER_FOLDER ${CMAKE_BINARY_DIR}/tools/clang/lib/DPCT/)
file(GLOB SRC_FILES ../../examples/DPCT/*/*.cu)
add_custom_target(gen_mapping_register
  COMMAND "${CMAKE_COMMAND}"
    -DTARGET="${API_MAPPING_REGISTER_FOLDER}APIMappingRegister.def"
    -DSRC_DIRECTORY="${CMAKE_CURRENT_SOURCE_DIR}/../../examples/DPCT/"
    -P ${CMAKE_CURRENT_SOURCE_DIR}/APIMapping/GenerateRegister.cmake
  DEPENDS ${SRC_FILES})
include_directories(
  ${API_MAPPING_REGISTER_FOLDER}
  )

include_directories(
  # For the CUDA-Toolchain and its CudaInstallationDetector
  ../../../clang/lib/Driver/
  )

add_subdirectory(ConfusableTable)

set(DPCT_SRC
  ASTTraversal.cpp
  AnalysisInfo.cpp
  Rewriters/CUB/CallExprRewriterCUB.cpp
  Rewriters/CUB/RewriterClassMethods.cpp
  Rewriters/CUB/RewriterDeviceHistgram.cpp
  Rewriters/CUB/RewriterDeviceMergeSort.cpp
  Rewriters/CUB/RewriterDeviceRadixSort.cpp
  Rewriters/CUB/RewriterDeviceReduce.cpp
  Rewriters/CUB/RewriterDeviceRunLengthEncode.cpp
  Rewriters/CUB/RewriterDeviceScan.cpp
  Rewriters/CUB/RewriterDeviceSegmentedRadixSort.cpp
  Rewriters/CUB/RewriterDeviceSegmentedReduce.cpp
  Rewriters/CUB/RewriterDeviceSegmentedSort.cpp
  Rewriters/CUB/RewriterDeviceSelect.cpp
  Rewriters/CUB/RewriterUtilityFunctions.cpp
  Rewriters/CUB/RewriterDevicePartition.cpp
  Rewriters/Math/CallExprRewriterMath.cpp
  Rewriters/Math/RewriterSTDFunctions.cpp
  Rewriters/Math/RewriterHalfArithmeticFunctions.cpp
  Rewriters/Math/RewriterHalf2ArithmeticFunctions.cpp
  Rewriters/Math/RewriterHalfComparisonFunctions.cpp
  Rewriters/Math/RewriterHalf2ComparisonFunctions.cpp
  Rewriters/Math/RewriterHalfPrecisionConversionAndDataMovement.cpp
  Rewriters/Math/RewriterHalfMathFunctions.cpp
  Rewriters/Math/RewriterHalf2MathFunctions.cpp
  Rewriters/Math/RewriterBfloat16PrecisionConversionAndDataMovement.cpp
  Rewriters/Math/RewriterSinglePrecisionMathematicalFunctions.cpp
  Rewriters/Math/RewriterDoublePrecisionMathematicalFunctions.cpp
  Rewriters/Math/RewriterIntegerMathematicalFunctions.cpp
  Rewriters/Math/RewriterSinglePrecisionIntrinsics.cpp
  Rewriters/Math/RewriterDoublePrecisionIntrinsics.cpp
  Rewriters/Math/RewriterIntegerIntrinsics.cpp
  Rewriters/Math/RewriterSIMDIntrinsics.cpp
  Rewriters/Math/RewriterCXXAPIRoutines.cpp
  Rewriters/Math/RewriterOverload.cpp
  CallExprRewriter.cpp
  CallExprRewriterAtomic.cpp
  CallExprRewriterCUFFT.cpp
  CallExprRewriterCUBLAS.cpp
  CallExprRewriterCURAND.cpp
  CallExprRewriterCUSOLVER.cpp
  CallExprRewriterCUSPARSE.cpp
  CallExprRewriterComplex.cpp
  CallExprRewriterDriver.cpp
  CallExprRewriterMemory.cpp
  CallExprRewriterMisc.cpp
  CallExprRewriterNccl.cpp
  CallExprRewriterStream.cpp
  CallExprRewriterTexture.cpp
  CallExprRewriterThrust.cpp
  CallExprRewriterWarp.cpp
  CallExprRewriterCUDNN.cpp
  CallExprRewriterErrorHandling.cpp
  CallExprRewriterLIBCU.cpp
  CallExprRewriterEvent.cpp
  CallExprRewriterCG.cpp
  CallExprRewriterWmma.cpp
  CrashRecovery.cpp
  Diagnostics.cpp
  Error.cpp
  Statics.cpp
  ExprAnalysis.cpp
  ExtReplacements.cpp
  MapNames.cpp
  SaveNewFiles.cpp
  DPCT.cpp
  TextModification.cpp
  Utility.cpp
  ValidateArguments.cpp
  ExternalReplacement.cpp
  VcxprojParser.cpp
  LibraryAPIMigration.cpp
  GenMakefile.cpp
  InclusionHeaders.cpp
  IncrementalMigrationUtility.cpp
  Rules.cpp
  PatternRewriter.cpp
  MigrateCmakeScript.cpp
  Homoglyph.cpp
  MisleadingBidirectional.cpp
  BarrierFenceSpaceAnalyzer.cpp
  BLASAPIMigration.cpp
  FFTAPIMigration.cpp
  DNNAPIMigration.cpp
  NCCLAPIMigration.cpp
  TypeLocRewriters.cpp
  AutoComplete.cpp
  AsmMigration.cpp
  APIMapping/QueryAPIMapping.cpp
  Asm/AsmLexer.cpp
  Asm/AsmParser.cpp
  Asm/AsmIdentifierTable.cpp
  Asm/AsmTokenKinds.cpp
  LIBCUAPIMigration.cpp
  CUBAPIMigration.cpp
  MemberExprRewriter.cpp
  MigrationRuleManager.cpp
  MigrationAction.cpp
  ThrustAPIMigration.cpp
  GenHelperFunction.cpp
  WMMAAPIMigration.cpp
  OptimizeMigration.cpp
  GroupFunctionAnalyzer.cpp
  SpBLASAPIMigration.cpp
  )

set(DPCT_TMP_DEPENDS
  ClangDriverOptions
  DeviceConfigFile
  dpct_helper_headers_and_inc
  genconfusable
  gen_mapping_register
  )

set(DPCT_LINK_LIBS
  clangBasic
  clangLex
  clangAnalysis
  clangAST
  clangASTMatchers
  clangDriver
  clangEdit
  clangFormat
  clangFrontend
  clangParse
  clangRewrite
  clangSema
  clangSerialization
  clangTooling
  clangToolingCore
  )

set(DPCT_DEPENDS ${DPCT_TMP_DEPENDS})

file(WRITE ${API_MAPPING_REGISTER_FOLDER}APIMappingRegisterBuffer.def "")
if (SYCLomatic_OPT_FOR_QUERY)
  add_clang_library(DPCT_TMP
    ${DPCT_SRC}

    DEPENDS
    ${DPCT_TMP_DEPENDS}

    LINK_LIBS
    ${DPCT_LINK_LIBS}
  )

  file(GLOB LIB_PATHS ../../examples/DPCT/*)
  list(REMOVE_ITEM LIB_PATHS "README.md")
  if(WIN32)
    list(REMOVE_ITEM LIB_PATHS "Thrust" "CUB")
  endif()
  foreach(LIB_PATH ${LIB_PATHS})
    string(REGEX REPLACE ".*/(.*)$" "\\1" LIB_NAME ${LIB_PATH})
    file(GLOB SRC_FILES RELATIVE ${LIB_PATH} ${LIB_PATH}/*.cu)
    list(LENGTH SRC_FILES API_NUM)
    set(BEGIN 0)
    set(STEP 50)
    while(${BEGIN} LESS ${API_NUM})
      list(SUBLIST SRC_FILES ${BEGIN} ${STEP} FILE_LIST)
      list(JOIN FILE_LIST "," FILE_LIST_STR)
      set(TARGET APIMappingRegisterBuffer${LIB_NAME}${BEGIN}.def)
      file(APPEND ${API_MAPPING_REGISTER_FOLDER}APIMappingRegisterBuffer.def "#include \"${TARGET}\"\n")
      file(WRITE ${API_MAPPING_REGISTER_FOLDER}${TARGET} "")
      add_custom_target(gen_mapping_str_${LIB_NAME}${BEGIN}
        COMMAND "${CMAKE_COMMAND}"
          -DTARGET="${API_MAPPING_REGISTER_FOLDER}${TARGET}"
          -DFILE_LIST_STR="${FILE_LIST_STR}"
          -P ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/DPCT/APIMapping/GenerateRegisterBuffer.cmake
        DEPENDS dpct-binary-tmp)
      list(APPEND DPCT_DEPENDS gen_mapping_str_${LIB_NAME}${BEGIN})
      math(EXPR BEGIN "${BEGIN} + ${STEP}")
    endwhile()
  endforeach()
endif()

add_clang_library(DPCT
  ${DPCT_SRC}

  DEPENDS
  ${DPCT_DEPENDS}

  LINK_LIBS
  ${DPCT_LINK_LIBS}
)
