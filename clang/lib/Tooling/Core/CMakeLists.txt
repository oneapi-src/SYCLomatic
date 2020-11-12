set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangToolingCore
  Diagnostic.cpp
  Replacement.cpp

  LINK_LIBS
  clangBasic
  clangLex
  clangRewrite
  )

# INTEL_CUSTOMIZATION
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DINTEL_CUSTOMIZATION=1" )
# end INTEL_CUSTOMIZATION
