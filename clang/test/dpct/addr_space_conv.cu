// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none --out-root %T/addr_space_conv %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --input-file %T/addr_space_conv/addr_space_conv.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -DNO_BUILD_TEST -fsycl %T/addr_space_conv/addr_space_conv.dp.cpp -o %T/addr_space_conv/addr_space_conv.dp.o %}

#include <cstdint>

__global__ void kernel1(const void* ptr) {
  // In PTX, addresses of the local and shared memory spaces are always 32 bits in size.
  __shared__ float shared_array[1024];
  // CHECK: auto smem = shared_array;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(shared_array));
#ifndef NO_BUILD_TEST
  asm volatile(
    "{\n"
    "   cp.async.cg.shared.global [%0], [%1], 16;\n"
    "}\n" :: "r"(smem), "l"(ptr)
  );
#endif
}

void foo(const void* ptr) {
  kernel1<<<1, 1>>>(ptr);
}
