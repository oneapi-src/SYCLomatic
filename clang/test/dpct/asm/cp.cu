// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/cp %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/cp/cp.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/cp/cp.dp.cpp -o %T/cp/cp.dp.o %}

// clang-format off
#include <cstdint>
#include <cstdint>
#include <cuda_runtime.h>

// CHECK: inline void cp_async4(void *smem_ptr, const void *glob_ptr) {
// CHECK-NEXT:  const int BYTES = 16;
// CHECK-NEXT:  auto smem = smem_ptr;
// CHECK-NEXT:  /*
// CHECK-NEXT:  DPCT1137:{{[0-9]+}}: ASM instruction "cp.async" is asynchronous copy, current it is migrated to synchronous copy operation. You may need to adjust the code to tune the performance.
// CHECK-NEXT:  */
// CHECK-NEXT:  {
// CHECK-NEXT:    *(((uint32_t *)(uintptr_t)smem)) = *(((uint32_t *)(uintptr_t)glob_ptr));
// CHECK-NEXT:    if (BYTES > 4)
// CHECK-NEXT:      *(((uint32_t *)(uintptr_t)smem) + 1) = *(((uint32_t *)(uintptr_t)glob_ptr) + 1);
// CHECK-NEXT:    if (BYTES > 8)
// CHECK-NEXT:      *(((uint32_t *)(uintptr_t)smem) + 2) = *(((uint32_t *)(uintptr_t)glob_ptr) + 2);
// CHECK-NEXT:    if (BYTES > 12)
// CHECK-NEXT:      *(((uint32_t *)(uintptr_t)smem) + 3) = *(((uint32_t *)(uintptr_t)glob_ptr) + 3);
// CHECK-NEXT:  }
// CHECK-NEXT:}
__device__ inline void cp_async4(void *smem_ptr, const void *glob_ptr) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("{\n"
               "   cp.async.cg.shared.global [%0], [%1], %2;\n"
               "}\n" :: "r"(smem), "l"(glob_ptr), "n"(BYTES));
}


// CHECK: inline void cp_async4_pred(void *smem_ptr, const void *glob_ptr,
// CHECK-NEXT:                                      bool pred = true) {
// CHECK-NEXT:  const int BYTES = 16;
// CHECK-NEXT:  auto smem = smem_ptr;
// CHECK-NEXT:  /*
// CHECK-NEXT:  DPCT1137:{{[0-9]+}}: ASM instruction "cp.async" is asynchronous copy, current it is migrated to synchronous copy operation. You may need to adjust the code to tune the performance.
// CHECK-NEXT:  */
// CHECK-NEXT:  {
// CHECK-NEXT:    bool p;
// CHECK-NEXT:    p = (int)pred != 0;
// CHECK-NEXT:    if (p) {
// CHECK-NEXT:      *(((uint32_t *)(uintptr_t)smem)) = *(((uint32_t *)(uintptr_t)glob_ptr));
// CHECK-NEXT:      if (BYTES > 4)
// CHECK-NEXT:        *(((uint32_t *)(uintptr_t)smem) + 1) = *(((uint32_t *)(uintptr_t)glob_ptr) + 1);
// CHECK-NEXT:      if (BYTES > 8)
// CHECK-NEXT:        *(((uint32_t *)(uintptr_t)smem) + 2) = *(((uint32_t *)(uintptr_t)glob_ptr) + 2);
// CHECK-NEXT:      if (BYTES > 12)
// CHECK-NEXT:        *(((uint32_t *)(uintptr_t)smem) + 3) = *(((uint32_t *)(uintptr_t)glob_ptr) + 3);
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:}
__device__ inline void cp_async4_pred(void *smem_ptr, const void *glob_ptr,
                                      bool pred = true) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("{\n"
               "   .reg .pred p;\n"
               "   setp.ne.b32 p, %0, 0;\n"
               "   @p cp.async.cg.shared.global [%1], [%2], %3;\n"
               "}\n" ::"r"((int)pred),
               "r"(smem), "l"(glob_ptr), "n"(BYTES));
}

// clang-format on
