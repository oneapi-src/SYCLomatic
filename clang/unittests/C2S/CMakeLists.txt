set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_unittest(C2STests
  SaveNewFilesTest.cpp
  ValidateArgumentsTest.cpp
  AnalysisInfoTest.cpp
  )

target_link_libraries(C2STests
  PRIVATE
  C2S
  clangBasic
  clangFrontend
  clangRewrite
  clangTooling
  clangToolingCore
  )
