// RUN: dpct --format-range=none -out-root %T/multi_placeholder_in_macro %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -ptx
// RUN: FileCheck %s --match-full-lines --input-file %T/multi_placeholder_in_macro/multi_placeholder_in_macro.dp.cpp

// CHECK: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
#include <cuda_runtime.h>

// CHECK: #define MACRO(PTR)
// CHECK-NEXT: do {
// CHECK-NEXT:   sycl::free(PTR, q_ct1);
// CHECK-NEXT:   sycl::free(PTR, q_ct1);
// CHECK-NEXT: } while (0)
#define MACRO(PTR)  \
do {                \
  cudaFree(PTR);    \
  cudaFree(PTR);    \
} while(0)

int main() {
  int *Ptr = nullptr;
  MACRO(Ptr);
  return 0;
}