// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/brkpt %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/brkpt/brkpt.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/brkpt/brkpt.dp.cpp -o %T/brkpt/brkpt.dp.o %}

// clang-format off
#include <cstdint>
#include <cuda_runtime.h>

// CHECK:inline void cp_async_wait_all() {
// CHECK-NEXT: /*
// CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to "brkpt;" was removed because this instruction is typically used for debugging. You may need to rewrite the code.
// CHECK-NEXT: */
// CHECK-NEXT:   // PTX breakpoint instruction
// CHECK-NEXT:}
__device__ inline void cp_async_wait_all() {
 asm("brkpt;");  // PTX breakpoint instruction
}

// clang-format on
