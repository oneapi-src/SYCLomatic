// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/prmt %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/prmt/prmt.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/prmt/prmt.dp.cpp -o %T/prmt/prmt.dp.o %}

// clang-format off
#include <cstdint>
#include <cuda_runtime.h>

__global__ void testKernel1(uint32_t *d_result, uint32_t a) {
  static constexpr uint32_t sel = 0x3210;
  static constexpr uint32_t b = 0;
  uint32_t d;

  // CHECK: d = dpct::byte_level_permute_custom(a, b, sel, 0);
  asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "n"(b), "n"(sel));

  // CHECK: d = dpct::byte_level_permute_custom(a, b, sel, 1);
  asm volatile("prmt.b32.f4e %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "n"(b), "n"(sel));

  // CHECK: d = dpct::byte_level_permute_custom(a, b, sel, 2);
  asm volatile("prmt.b32.b4e %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "n"(b), "n"(sel));

  // CHECK: d = dpct::byte_level_permute_custom(a, b, sel, 3);
  asm volatile("prmt.b32.rc8 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "n"(b), "n"(sel));

  // CHECK: d = dpct::byte_level_permute_custom(a, b, sel, 4);
  asm volatile("prmt.b32.ecl  %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "n"(b), "n"(sel));

  // CHECK: d = dpct::byte_level_permute_custom(a, b, sel, 5);
  asm volatile("prmt.b32.ecr %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "n"(b), "n"(sel));

  // CHECK: d = dpct::byte_level_permute_custom(a, b, sel, 6);
  asm volatile("prmt.b32.rc16 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "n"(b), "n"(sel));
}

// clang-format on
