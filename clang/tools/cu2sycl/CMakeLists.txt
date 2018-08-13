set(LLVM_LINK_COMPONENTS
  support
  )

add_clang_tool(cu2sycl
  Cu2Sycl.cpp
  )

target_link_libraries(cu2sycl
  PRIVATE
  Cu2Sycl
  )
