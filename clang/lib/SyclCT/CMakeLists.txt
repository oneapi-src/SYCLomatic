set(LLVM_LINK_COMPONENTS
  Option
  Support
  )

include_directories(
  # For the CUDA-Toolchain and its CudaInstallationDetector
  ../../../clang/lib/Driver/
  )

add_clang_library(SyclCT
  ASTTraversal.cpp
  AnalysisInfo.cpp
  Debug.cpp
  Diagnostics.cpp
  ExprAnalysis.cpp
  MapNames.cpp
  SaveNewFiles.cpp
  SyclCT.cpp
  TextModification.cpp
  Utility.cpp
  ValidateArguments.cpp

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
