// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct -in-root %S -out-root %T/intrinsic/iadd3 %S/iadd3.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/intrinsic/iadd3/iadd3.dp.cpp --match-full-lines %s

// CHECK:#include <sycl/sycl.hpp>
// CHECK:#include <dpct/dpct.hpp>
#include <cub/cub.cuh>
#include <limits>
#include <stdio.h>

// CHECK:void kernel(int *res) {
// CHECK-NEXT:  *res = static_cast<unsigned>(1 + 2 + 3);
// CHECK-NEXT:}
__global__ void kernel(int *res) {
  *res = cub::IADD3(1, 2, 3);
}

int main() {
  int *val = nullptr;
  int v = 0;
  cudaMalloc(&val, sizeof(int));
  kernel<<<1, 1>>>(val);
  cudaMemcpy(&v, val, sizeof(int), cudaMemcpyDeviceToHost);
  printf("%d\n", v);
  return 0;
}
