// RUN: dpct --format-range=none -out-root %T/multi_placeholder_in_macro %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -ptx
// RUN: FileCheck %s --input-file %T/multi_placeholder_in_macro/multi_placeholder_in_macro.dp.cpp

// CHECK: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
#include <cuda_runtime.h>

// CHECK: #define MACRO(PTR, SIZE) 
// CHECK-NEXT:  do {
// CHECK-NEXT:     PTR = (int *)sycl::malloc_device(SIZE, q_ct1);
// CHECK-NEXT:     sycl::free(PTR, dpct::get_default_queue());
// CHECK-NEXT:     sycl::free(PTR, dpct::get_default_queue());
// CHECK-NEXT:     sycl::free(PTR, dpct::get_default_queue());
// CHECK-NEXT:   } while (0);
#define MACRO(PTR, SIZE)    \
  do {                      \
    cudaMalloc(&PTR, SIZE); \
    cudaFree(PTR);\
    cudaFree(PTR);\
    cudaFree(PTR);\
  } while (0);

int main() {
  int *Ptr = nullptr;
  MACRO(Ptr, 8);
  return 0;
}
