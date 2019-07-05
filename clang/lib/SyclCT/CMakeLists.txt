set(LLVM_LINK_COMPONENTS
  Option
  Support
  )
include_directories(
  # For the CUDA-Toolchain and its CudaInstallationDetector
  ../../../clang/lib/Driver/
  GaHelper/
  LibcurlLinux/include/
  )
if(UNIX)
  set(LIBCURL ${CMAKE_SOURCE_DIR}/../clang/lib/SyclCT/LibcurlLinux/lib/libcurl.a)
else()
  set(LIBCURL "")
endif()


add_clang_library(SyclCT
  ASTTraversal.cpp
  AnalysisInfo.cpp
  CallExprRewriter.cpp
  Debug.cpp
  Diagnostics.cpp
  ExprAnalysis.cpp
  ExtReplacements.cpp
  MapNames.cpp
  SaveNewFiles.cpp
  SyclCT.cpp
  TextModification.cpp
  Utility.cpp
  ValidateArguments.cpp
  ExternalReplacement.cpp
  SignalProcess.cpp
  GaHelper/active_user_detector.cpp
  GaHelper/filelock.cpp
  GaHelper/filesystem_util.cpp
  GaHelper/gahelper_impl.cpp
  GaHelper/http_connector.cpp
  GaHelper/network_util.cpp
  GaHelper/os_specific.cpp
  GaHelper/uuid.cpp
  GAnalytics.cpp

  DEPENDS
  ClangDriverOptions

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
