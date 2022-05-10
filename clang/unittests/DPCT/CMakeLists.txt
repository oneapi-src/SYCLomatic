set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_unittest(DpctTests
  SaveNewFilesTest.cpp
  ValidateArgumentsTest.cpp
  AnalysisInfoTest.cpp
  )

target_link_libraries(DpctTests
  PRIVATE
  DPCT
  clangBasic
  clangFrontend
  clangRewrite
  clangTooling
  clangToolingCore
  )
