set(LLVM_LINK_COMPONENTS
  Option
  Support
  )

set(CONT_FILES
  ${CMAKE_SOURCE_DIR}/../clang/runtime/c2s-rt/include/atomic.hpp.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/c2s-rt/include/device.hpp.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/c2s-rt/include/image.hpp.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/c2s-rt/include/memory.hpp.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/c2s-rt/include/util.hpp.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/c2s-rt/include/blas_utils.hpp.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/c2s-rt/include/c2s.hpp.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/c2s-rt/include/dpl_utils.hpp.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/c2s-rt/include/kernel.hpp.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/c2s-rt/include/rng_utils.hpp.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/c2s-rt/include/lib_common_utils.hpp.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/c2s-rt/include/dpl_extras/algorithm.h.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/c2s-rt/include/dpl_extras/functional.h.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/c2s-rt/include/dpl_extras/iterators.h.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/c2s-rt/include/dpl_extras/memory.h.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/c2s-rt/include/dpl_extras/numeric.h.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/c2s-rt/include/dpl_extras/vector.h.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/c2s-rt/include/dpl_extras/dpcpp_extensions.h.inc
  )

set(PROCESS_FILES_OUTPUT
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/C2S/dpl_extras/algorithm.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/C2S/dpl_extras/dpcpp_extensions.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/C2S/dpl_extras/functional.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/C2S/dpl_extras/iterators.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/C2S/dpl_extras/memory.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/C2S/dpl_extras/numeric.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/C2S/dpl_extras/vector.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/C2S/dpl_extras/algorithm.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/C2S/dpl_extras/dpcpp_extensions.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/C2S/dpl_extras/functional.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/C2S/dpl_extras/iterators.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/C2S/dpl_extras/memory.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/C2S/dpl_extras/numeric.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/C2S/dpl_extras/vector.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/C2S/atomic.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/C2S/blas_utils.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/C2S/device.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/C2S/c2s.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/C2S/dpl_utils.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/C2S/image.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/C2S/kernel.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/C2S/memory.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/C2S/util.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/C2S/rng_utils.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/C2S/lib_common_utils.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/C2S/atomic.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/C2S/blas_utils.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/C2S/device.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/C2S/c2s.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/C2S/dpl_utils.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/C2S/image.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/C2S/kernel.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/C2S/memory.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/C2S/util.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/C2S/rng_utils.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/C2S/lib_common_utils.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/C2S/HelperFeatureEnum.inc
)

add_custom_command(
  OUTPUT Confusables.inc
  COMMAND make_confusable_table ${CMAKE_CURRENT_SOURCE_DIR}/ConfusableTable/confusables.txt ${CMAKE_CURRENT_BINARY_DIR}/Confusables.inc
  DEPENDS make_confusable_table ConfusableTable/confusables.txt
  )

add_custom_command(
  DEPENDS ${CONT_FILES} ${CMAKE_SOURCE_DIR}/../clang/runtime/c2s-rt/include/HelperFileAndFeatureNames.inc ${CMAKE_SOURCE_DIR}/../clang/runtime/c2s-rt/include/processFiles.py
  OUTPUT  ${PROCESS_FILES_OUTPUT}
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/../clang/runtime/c2s-rt/include/processFiles.py --build-dir ${CMAKE_BINARY_DIR}
  )

add_custom_target(genconfusable 
  DEPENDS Confusables.inc
  )

add_custom_target(c2s_helper_headers_and_inc
  DEPENDS ${PROCESS_FILES_OUTPUT}
  )

include_directories(
  # For the CUDA-Toolchain and its CudaInstallationDetector
  ../../../clang/lib/Driver/
  GaHelper/
  libcurl/include/
  )
if(UNIX)
  set(LIBCURL ${CMAKE_SOURCE_DIR}/../clang/lib/C2S/libcurl/lib/linux/libcurl.a)
else()
  set(LIBCURL ${CMAKE_SOURCE_DIR}/../clang/lib/C2S/libcurl/lib/win/libcurl_a.lib)
endif()

add_subdirectory(docs)
add_subdirectory(ConfusableTable)

add_clang_library(C2S
  ASTTraversal.cpp
  AnalysisInfo.cpp
  CallExprRewriter.cpp
  Diagnostics.cpp
  Error.cpp
  Statics.cpp
  ExprAnalysis.cpp
  ExtReplacements.cpp
  MapNames.cpp
  SaveNewFiles.cpp
  C2S.cpp
  TextModification.cpp
  Utility.cpp
  ValidateArguments.cpp
  ExternalReplacement.cpp
  SignalProcess.cpp
  VcxprojParser.cpp
  GaHelper/active_user_detector.cpp
  GaHelper/filelock.cpp
  GaHelper/filesystem_util.cpp
  GaHelper/gahelper_impl.cpp
  GaHelper/http_connector.cpp
  GaHelper/network_util.cpp
  GaHelper/os_specific.cpp
  GaHelper/uuid.cpp
  GAnalytics.cpp
  LibraryAPIMigration.cpp
  CustomHelperFiles.cpp
  GenMakefile.cpp
  IncrementalMigrationUtility.cpp
  Rules.cpp
  Homoglyph.cpp
  MisleadingBidirectional.cpp
  BarrierFenceSpaceAnalyzer.cpp

  DEPENDS
  ClangDriverOptions
  c2s_helper_headers_and_inc
  genconfusable

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
  ${LIBCURL}
  )