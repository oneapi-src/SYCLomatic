set(LLVM_LINK_COMPONENTS
  Option
  Support
  )

set(CONT_FILES
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/atomic.hpp.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/device.hpp.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/image.hpp.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/memory.hpp.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/util.hpp.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/blas_utils.hpp.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpct.hpp.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpl_utils.hpp.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/kernel.hpp.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpl_extras/algorithm.h.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpl_extras/functional.h.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpl_extras/iterators.h.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpl_extras/memory.h.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpl_extras/numeric.h.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpl_extras/vector.h.inc
  ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/dpl_extras/dpcpp_extensions.h.inc
  )

set(PROCESS_FILES_OUTPUT
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dpl_extras/algorithm.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dpl_extras/dpcpp_extensions.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dpl_extras/functional.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dpl_extras/iterators.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dpl_extras/memory.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dpl_extras/numeric.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dpl_extras/vector.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dpl_extras/algorithm.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dpl_extras/dpcpp_extensions.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dpl_extras/functional.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dpl_extras/iterators.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dpl_extras/memory.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dpl_extras/numeric.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dpl_extras/vector.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/atomic.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/blas_utils.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/device.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dpct.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dpl_utils.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/image.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/kernel.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/memory.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/util.all.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/atomic.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/blas_utils.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/device.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dpct.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/dpl_utils.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/image.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/kernel.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/memory.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/util.inc
  ${CMAKE_BINARY_DIR}/tools/clang/include/clang/DPCT/HelperFeatureEnum.inc
)

add_custom_command(
  DEPENDS ${CONT_FILES} ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/HelperFileAndFeatureNames.inc ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/processFiles.py
  OUTPUT  ${PROCESS_FILES_OUTPUT}
  COMMAND python3 ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/processFiles.py --build-dir ${CMAKE_BINARY_DIR}
  )

add_custom_target(dpct_helper_headers_and_inc
  DEPENDS ${PROCESS_FILES_OUTPUT}
  )

include_directories(
  # For the CUDA-Toolchain and its CudaInstallationDetector
  ../../../clang/lib/Driver/
  GaHelper/
  libcurl/include/
  )
if(UNIX)
  set(LIBCURL ${CMAKE_SOURCE_DIR}/../clang/lib/DPCT/libcurl/lib/linux/libcurl.a)
else()
  set(LIBCURL ${CMAKE_SOURCE_DIR}/../clang/lib/DPCT/libcurl/lib/win/libcurl_a.lib)
endif()

add_subdirectory(docs)

add_clang_library(DPCT
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
  DPCT.cpp
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

  DEPENDS
  ClangDriverOptions
  dpct_helper_headers_and_inc

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
