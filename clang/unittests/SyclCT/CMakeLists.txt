set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_unittest(SyclCTTests
  SaveNewFilesTest.cpp
  ValidateArgumentsTest.cpp
  )

target_link_libraries(SyclCTTests
  PRIVATE
  SyclCT
  clangBasic
  clangFrontend
  clangRewrite
  clangTooling
  clangToolingCore
  )
