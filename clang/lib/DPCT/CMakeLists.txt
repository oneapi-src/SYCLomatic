set(LLVM_LINK_COMPONENTS
  Option
  Support
  )

set(RUNTIME_HEADERS
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/atomic.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/device.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/image.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/math.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/memory.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/util.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/blas_utils.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dnnl_utils.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpct.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpl_utils.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/kernel.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/rng_utils.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/lib_common_utils.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/ccl_utils.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/sparse_utils.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/fft_utils.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/lapack_utils.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpl_extras/algorithm.h
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpl_extras/functional.h
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpl_extras/iterators.h
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpl_extras/memory.h
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpl_extras/numeric.h
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpl_extras/vector.h
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpl_extras/dpcpp_extensions.h
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
set(API_MAPPING_REGISTER_FOLDER ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/)
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

add_clang_library(DPCT
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
  CallExprRewriterMath.cpp
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

  DEPENDS
  ClangDriverOptions
  DeviceConfigFile
  dpct_helper_headers_and_inc
  genconfusable
  gen_mapping_register

  LINK_LIBS
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
add_subdirectory(docs)
