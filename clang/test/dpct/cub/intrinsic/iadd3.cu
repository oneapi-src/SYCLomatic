// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct -in-root %S -out-root %T/intrinsic/iadd3 %S/iadd3.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/intrinsic/iadd3/iadd3.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/intrinsic/iadd3/iadd3.dp.cpp -o %T/intrinsic/iadd3/iadd3.dp.o %}

// CHECK:#include <sycl/sycl.hpp>
// CHECK:#include <dpct/dpct.hpp>
#include <cub/cub.cuh>
#include <limits>
#include <stdio.h>

// CHECK:void kernel1(int *res) {
// CHECK-NEXT:  *res = ((unsigned int)1 + (unsigned int)2 + (unsigned int)3);
// CHECK-NEXT:}
__global__ void kernel1(int *res) {
  *res = cub::IADD3(1, 2, 3);
}

// CHECK:void kernel2(int *res, unsigned a, int b, unsigned c) {
// CHECK-NEXT:  *res = (a + (unsigned int)b + c);
// CHECK-NEXT:}
__global__ void kernel2(int *res, unsigned a, int b, unsigned c) {
  *res = cub::IADD3(a, b, c);
}

// CHECK:void kernel3(int *res, unsigned a, unsigned b, unsigned c) {
// CHECK-NEXT:  *res = (a + b + c);
// CHECK-NEXT:}
__global__ void kernel3(int *res, unsigned a, unsigned b, unsigned c) {
  *res = cub::IADD3(a, b, c);
}

int main() {
  int *val = nullptr;
  int v = 0;
  cudaMalloc(&val, sizeof(int));
  kernel1<<<1, 1>>>(val);
  cudaMemcpy(&v, val, sizeof(int), cudaMemcpyDeviceToHost);
  printf("%d\n", v);
  return 0;
}
