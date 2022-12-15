set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ")
set(LLVM_LINK_COMPONENTS
  support
)

add_clang_tool(pattern-rewriter-binary
  src/driver.cpp
  src/pattern.cpp
)
set_target_properties(pattern-rewriter-binary PROPERTIES OUTPUT_NAME "pattern-rewriter")
add_llvm_install_targets("install-dpct-pattern-rewriter"
                         DEPENDS pattern-rewriter-binary
                         COMPONENT pattern-rewriter-binary
)
