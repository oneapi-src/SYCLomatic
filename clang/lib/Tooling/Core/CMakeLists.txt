set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangToolingCore
  Diagnostic.cpp
  Replacement.cpp

  LINK_LIBS
  clangBasic
  clangLex
  clangRewrite
  )

# SYCLomatic_CUSTOMIZATION begin
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSYCLomatic_CUSTOMIZATION=1" )
# SYCLomatic_CUSTOMIZATION end
