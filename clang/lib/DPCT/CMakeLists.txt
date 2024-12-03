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
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpct/detail/group_utils_detail.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpct/detail/math_detail.hpp
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpct/detail/memory_detail.hpp
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
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/detail/group_utils_detail.hpp.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/detail/math_detail.hpp.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/detail/memory_detail.hpp.inc
)

add_custom_command(
  OUTPUT Confusables.inc
  COMMAND make_confusable_table ${CMAKE_CURRENT_SOURCE_DIR}/RulesSecurity/ConfusableTable/confusables.txt ${CMAKE_CURRENT_BINARY_DIR}/Confusables.inc
  DEPENDS make_confusable_table RulesSecurity/ConfusableTable/confusables.txt
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
    -P ${CMAKE_CURRENT_SOURCE_DIR}/QueryAPIMapping/GenerateRegister.cmake
  DEPENDS ${SRC_FILES})
include_directories(
  ${API_MAPPING_REGISTER_FOLDER}
  )

include_directories(
  # For the CUDA-Toolchain and its CudaInstallationDetector
  ../../../clang/lib/Driver/
  )

add_subdirectory(RulesSecurity/ConfusableTable)

add_clang_library(DPCT
  ASTTraversal.cpp
  PreProcessor.cpp
  AnalysisInfo.cpp
  RulesLang/RulesLang.cpp
  RulesLang/RulesLangGraph.cpp
  RulesLang/RulesLangGraphInterop.cpp
  RulesLang/RulesLangNoneAPIAndType.cpp
  RulesLang/RulesLangAtomic.cpp
  RulesLang/RulesLangCooperativeGroups.cpp
  RulesLang/RulesLangTexture.cpp
  RulesLang/RulesLangAddrSpaceConv.cpp
  RulesLangLib/CUB/CallExprRewriterCUB.cpp
  RulesLangLib/CUB/RewriterClassMethods.cpp
  RulesLangLib/CUB/RewriterDeviceHistgram.cpp
  RulesLangLib/CUB/RewriterDeviceMergeSort.cpp
  RulesLangLib/CUB/RewriterDeviceRadixSort.cpp
  RulesLangLib/CUB/RewriterDeviceReduce.cpp
  RulesLangLib/CUB/RewriterDeviceRunLengthEncode.cpp
  RulesLangLib/CUB/RewriterDeviceScan.cpp
  RulesLangLib/CUB/RewriterDeviceSegmentedRadixSort.cpp
  RulesLangLib/CUB/RewriterDeviceSegmentedReduce.cpp
  RulesLangLib/CUB/RewriterDeviceSegmentedSort.cpp
  RulesLangLib/CUB/RewriterDeviceSelect.cpp
  RulesLangLib/CUB/RewriterUtilityFunctions.cpp
  RulesLangLib/CUB/RewriterDevicePartition.cpp
  RulesLangLib/CUB/RewriterDeviceSpmv.cpp
  RulesLang/Math/CallExprRewriterMath.cpp
  RulesLang/Math/RewriterSTDFunctions.cpp
  RulesLang/Math/RewriterHalfArithmeticFunctions.cpp
  RulesLang/Math/RewriterHalf2ArithmeticFunctions.cpp
  RulesLang/Math/RewriterHalfComparisonFunctions.cpp
  RulesLang/Math/RewriterHalf2ComparisonFunctions.cpp
  RulesLang/Math/RewriterHalfPrecisionConversionAndDataMovement.cpp
  RulesLang/Math/RewriterHalfMathFunctions.cpp
  RulesLang/Math/RewriterHalf2MathFunctions.cpp
  RulesLang/Math/RewriterBfloat16PrecisionConversionAndDataMovement.cpp
  RulesLang/Math/RewriterSinglePrecisionMathematicalFunctions.cpp
  RulesLang/Math/RewriterDoublePrecisionMathematicalFunctions.cpp
  RulesLang/Math/RewriterIntegerMathematicalFunctions.cpp
  RulesLang/Math/RewriterSinglePrecisionIntrinsics.cpp
  RulesLang/Math/RewriterDoublePrecisionIntrinsics.cpp
  RulesLang/Math/RewriterIntegerIntrinsics.cpp
  RulesLang/Math/RewriterSIMDIntrinsics.cpp
  RulesLang/Math/RewriterCXXAPIRoutines.cpp
  RulesLang/Math/RewriterOverload.cpp
  RulesLang/RewriterSYCLcompat.cpp
  RuleInfra/CallExprRewriter.cpp
  RulesLang/CallExprRewriterAtomic.cpp
  RulesMathLib/CallExprRewriterCUFFT.cpp
  RulesMathLib/CallExprRewriterCUBLAS.cpp
  RulesMathLib/CallExprRewriterCURAND.cpp
  RulesMathLib/CallExprRewriterCUSOLVER.cpp
  RulesMathLib/CallExprRewriterCUSPARSE.cpp
  RulesLang/CallExprRewriterComplex.cpp
  RulesLang/CallExprRewriterDriver.cpp
  RulesLang/CallExprRewriterGraph.cpp
  RulesLang/CallExprRewriterGraphicsInterop.cpp
  RulesLang/CallExprRewriterMemory.cpp
  RulesLang/CallExprRewriterMisc.cpp
  RulesCCL/CallExprRewriterNccl.cpp
  RulesLang/CallExprRewriterStream.cpp
  RulesLang/CallExprRewriterTexture.cpp
  RulesLangLib/CallExprRewriterThrust.cpp
  RulesLang/CallExprRewriterWarp.cpp
  RulesDNN/CallExprRewriterCUDNN.cpp
  RulesLang/CallExprRewriterErrorHandling.cpp
  RulesLangLib/CallExprRewriterLIBCU.cpp
  RulesLang/CallExprRewriterEvent.cpp
  RulesLang/CallExprRewriterCG.cpp
  RulesLang/CallExprRewriterWmma.cpp
  ErrorHandle/CrashRecovery.cpp
  Diagnostics/Diagnostics.cpp
  ErrorHandle/Error.cpp
  MigrationReport/Statics.cpp
  RuleInfra/ExprAnalysis.cpp
  ExtReplacements.cpp
  RuleInfra/MapNames.cpp
  RuleInfra/MigrationStatistics.cpp
  RulesMathLib/MapNamesSolver.cpp
  RulesMathLib/MapNamesBlas.cpp
  RulesMathLib/MapNamesRandom.cpp
  RulesLang/MapNamesLang.cpp
  RulesDNN/MapNamesDNN.cpp
  RulesLangLib/MapNamesLangLib.cpp
  FileGenerator/GenFiles.cpp
  DPCT.cpp
  CommandOption/DpctOptions.cpp
  TextModification.cpp
  Utility.cpp
  CommandOption/ValidateArguments.cpp
  IncMigration/ExternalReplacement.cpp
  Windows/VcxprojParser.cpp
  MigrateScript/GenMakefile.cpp
  RulesInclude/InclusionHeaders.cpp
  IncMigration/IncrementalMigrationUtility.cpp
  UserDefinedRules/UserDefinedRules.cpp
  UserDefinedRules/PatternRewriter.cpp
  MigrateScript/MigrateBuildScript.cpp
  MigrateScript/MigrateCmakeScript.cpp
  MigrateScript/MigratePythonBuildScript.cpp
  RulesSecurity/Homoglyph.cpp
  RulesSecurity/MisleadingBidirectional.cpp
  RulesLang/BarrierFenceSpaceAnalyzer.cpp
  RulesMathLib/BLASAPIMigration.cpp
  RulesMathLib/FFTAPIMigration.cpp
  RulesDNN/DNNAPIMigration.cpp
  RulesCCL/NCCLAPIMigration.cpp
  RuleInfra/TypeLocRewriters.cpp
  Linux/AutoComplete.cpp
  RulesAsm/AsmMigration.cpp
  QueryAPIMapping/QueryAPIMapping.cpp
  RulesAsm/Parser/AsmNodes.cpp
  RulesAsm/Parser/AsmLexer.cpp
  RulesAsm/Parser/AsmParser.cpp
  RulesAsm/Parser/AsmIdentifierTable.cpp
  RulesAsm/Parser/AsmTokenKinds.cpp
  RulesLangLib/LIBCUAPIMigration.cpp
  RulesLangLib/CUBAPIMigration.cpp
  RuleInfra/MemberExprRewriter.cpp
  MigrationRuleManager.cpp
  MigrationAction.cpp
  RulesLangLib/ThrustAPIMigration.cpp
  FileGenerator/GenHelperFunction.cpp
  RulesLang/WMMAAPIMigration.cpp
  RulesLang/OptimizeMigration.cpp
  RulesLang/GroupFunctionAnalyzer.cpp
  RulesMathLib/SpBLASAPIMigration.cpp
  RulesMathLib/RandomAPIMigration.cpp
  RulesMathLib/SolverAPIMigration.cpp
  CodePin/GenCodePinHeader.cpp

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
