set(LLVM_LINK_COMPONENTS
  support
  )

add_clang_tool(cu2sycl
  Cu2Sycl.cpp
  SourceTransformation.cpp
  Translation.cpp
  )

target_link_libraries(cu2sycl
  PRIVATE
  clangAnalysis
  clangAST
  clangASTMatchers
  clangBasic
  clangDriver
  clangEdit
  clangFormat
  clangFrontend
  clangLex
  clangParse
  clangRewrite
  clangSema
  clangSerialization
  clangTooling
  clangToolingCore
  )
