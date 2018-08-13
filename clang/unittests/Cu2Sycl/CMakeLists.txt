set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_unittest(Cu2SyclTests
  SaveNewFilesTest.cpp
  ValidateArgumentsTest.cpp
  )
target_link_libraries(Cu2SyclTests
  PRIVATE
  Cu2Sycl
  )
