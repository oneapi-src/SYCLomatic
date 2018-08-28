set(LLVM_LINK_COMPONENTS
  Option
  Support
  )

include_directories(
  # For the CUDA-Toolchain and its CudaInstallationDetector
  ../../../clang/lib/Driver/
  )

add_clang_library(SyclCT
  SyclCT.cpp
  ASTTraversal.cpp
  MapNames.cpp
  SaveNewFiles.cpp
  TextModification.cpp
  Utility.cpp
  ValidateArguments.cpp
  Diagnostics.cpp

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
