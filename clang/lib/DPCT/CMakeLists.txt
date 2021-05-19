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
  )

add_custom_target(dpct_helper_headers_and_inc
  COMMAND python3 ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/processFiles.py --build-dir ${CMAKE_BINARY_DIR}
  DEPENDS ${CONT_FILES} ${CMAKE_SOURCE_DIR}/../clang/runtime/dpct-rt/include/processFiles.py
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
  Debug.cpp
  Diagnostics.cpp
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
