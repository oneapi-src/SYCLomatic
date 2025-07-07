set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ")
set(LLVM_LINK_COMPONENTS
  support
)

add_clang_tool(gitdiff2yaml-binary
  src/main.cpp
  src/gitdiff2yaml.cpp
)
set_target_properties(gitdiff2yaml-binary PROPERTIES OUTPUT_NAME "gitdiff2yaml")
add_llvm_install_targets("install-dpct-gitdiff2yaml"
                         DEPENDS gitdiff2yaml-binary
                         COMPONENT gitdiff2yaml-binary
)
