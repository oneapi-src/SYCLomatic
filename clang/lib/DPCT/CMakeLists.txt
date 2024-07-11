set(LLVM_LINK_COMPONENTS
  Option
  Support
  )

set(RUNTIME_HEADERS
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpct/atomic.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpct/bindless_images.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpct/device.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpct/image.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpct/graph.hpp
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
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpct/group_utils.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpct/blas_gemm_utils.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpct/compat_service.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpct/dpl_extras/algorithm.h
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpct/dpl_extras/functional.h
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpct/dpl_extras/iterators.h
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpct/dpl_extras/iterator_adaptor.h
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
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/graph.hpp.inc
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
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/group_utils.hpp.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/blas_gemm_utils.hpp.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/compat_service.hpp.inc
)

add_custom_command(
  OUTPUT Confusables.inc
  COMMAND make_confusable_table ${CMAKE_CURRENT_SOURCE_DIR}/ConfusableTable/confusables.txt ${CMAKE_CURRENT_BINARY_DIR}/Confusables.inc
  DEPENDS make_confusable_table ConfusableTable/confusables.txt
  )

add_custom_command(
  DEPENDS ${RUNTIME_HEADERS} ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/processFiles.py  ${CMAKE_SOURCE_DIR}/../clang/tools/dpct/cmake/dpct.cmake
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
  Rewriters/CUB/RewriterDevicePartition.cpp
  Rewriters/CUB/RewriterDeviceSpmv.cpp
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
  Rewriters/RewriterSYCLcompat.cpp
  CallExprRewriter.cpp
  CallExprRewriterAtomic.cpp
  CallExprRewriterCUFFT.cpp
  CallExprRewriterCUBLAS.cpp
  CallExprRewriterCURAND.cpp
  CallExprRewriterCUSOLVER.cpp
  CallExprRewriterCUSPARSE.cpp
  CallExprRewriterComplex.cpp
  CallExprRewriterDriver.cpp
  CallExprRewriterGraph.cpp
  CallExprRewriterGraphicsInterop.cpp
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
  DpctOptions.cpp
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
  MigratePythonSetupScript.cpp
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
  Asm/AsmNodes.cpp
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
  GenCodePinHeader.cpp

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
