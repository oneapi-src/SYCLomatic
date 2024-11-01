// RUN: dpct --format-range=none --enable-codepin -out-root %T/debug_test/struct_kernel_call_in_same_namespace %s --cuda-include-path="%cuda-path/include" -- -std=c++17  -x cuda --cuda-host-only
// RUN: FileCheck %S/codepin_autogen_util.hpp.ref --match-full-lines --input-file %T/debug_test/struct_kernel_call_in_same_namespace_codepin_sycl/codepin_autogen_util.hpp
// RUN: FileCheck %S/codepin_autogen_util.hpp.cuda.ref --match-full-lines --input-file %T/debug_test/struct_kernel_call_in_same_namespace_codepin_cuda/codepin_autogen_util.hpp
// RUN: FileCheck %s --match-full-lines --input-file %T/debug_test/struct_kernel_call_in_same_namespace_codepin_sycl/test.dp.cpp
// RUN: FileCheck %s --match-full-lines --input-file %T/debug_test/struct_kernel_call_in_same_namespace_codepin_cuda/test.cu
// RUN: %if build_lit %{icpx -c -fsycl %T/debug_test/struct_kernel_call_in_same_namespace_codepin_sycl/test.dp.cpp -o %T/debug_test/struct_kernel_call_in_same_namespace_codepin_sycl/test.dp.o %}
#include <cuda.h>
#include <iostream>
namespace test {
struct P2 {
  int x;
  int y;
};
} // namespace test


struct CCC2 {
  int x;
  int y;
};

namespace test_codepin {
using Point2D = test::P2;
};

//CHECK:  namespace nnn {
namespace nnn {
struct PP2 {
  int x;
  int y;
};

using INT = int;

__global__ void kernel2d(test_codepin::Point2D *a, test_codepin::Point2D *b, test_codepin::Point2D *c) {
  int i = threadIdx.x;
  c[i].x = a[i].x + b[i].x;
  c[i].y = a[i].y + b[i].y;
}

__global__ void kernel2d_org(test::P2 *a, test::P2 *b, test::P2 *c) {
  int i = threadIdx.x;
  c[i].x = a[i].x + b[i].x;
  c[i].y = a[i].y + b[i].y;
}

__global__ void kernel2d_2(PP2 *a, PP2 *b, PP2 *c) {
  int i = threadIdx.x;
  c[i].x = a[i].x + b[i].x;
  c[i].y = a[i].y + b[i].y;
}

__global__ void kerneel_int(INT *a, INT *b, INT *c) {
  int i = threadIdx.x;
  c[i] = a[i] + b[i];
}

#define NUM 10
void test() {
  test_codepin::Point2D h_2d[NUM];
  for (int i = 0; i < NUM; i++) {
    h_2d[i].x = i;
    h_2d[i].y = i;
  }
  test_codepin::Point2D *d_a2d, *d_b2d, *d_c2d;
  cudaMalloc(&d_a2d, sizeof(test_codepin::Point2D) * NUM);
  cudaMalloc(&d_b2d, sizeof(test_codepin::Point2D) * NUM);
  cudaMalloc(&d_c2d, sizeof(test_codepin::Point2D) * NUM);
  cudaMemcpy(d_a2d, h_2d, sizeof(test_codepin::Point2D) * NUM, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b2d, h_2d, sizeof(test_codepin::Point2D) * NUM, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c2d, h_2d, sizeof(test_codepin::Point2D) * NUM, cudaMemcpyHostToDevice);
  kernel2d<<<1, NUM>>>(d_a2d, d_b2d, d_c2d);
  kernel2d_org<<<1, NUM>>>(d_a2d, d_b2d, d_c2d);
  cudaDeviceSynchronize();

  PP2 h_pp2_2d[NUM];
  for (int i = 0; i < NUM; i++) {
    h_pp2_2d[i].x = i;
    h_pp2_2d[i].y = i;
  }
  PP2 *d_pp2_a2d, *d_pp2_b2d, *d_pp2_c2d;
  cudaMalloc(&d_pp2_a2d, sizeof(PP2) * NUM);
  cudaMalloc(&d_pp2_b2d, sizeof(PP2) * NUM);
  cudaMalloc(&d_pp2_c2d, sizeof(PP2) * NUM);
  cudaMemcpy(d_pp2_a2d, h_pp2_2d, sizeof(PP2) * NUM, cudaMemcpyHostToDevice);
  cudaMemcpy(d_pp2_b2d, h_pp2_2d, sizeof(PP2) * NUM, cudaMemcpyHostToDevice);
  cudaMemcpy(d_pp2_c2d, h_pp2_2d, sizeof(PP2) * NUM, cudaMemcpyHostToDevice);
  kernel2d_2<<<1, NUM>>>(d_pp2_a2d, d_pp2_b2d, d_pp2_c2d);
  cudaDeviceSynchronize();

  INT h_int_2d[NUM];
  for (int i = 0; i < NUM; i++) {
    h_int_2d[i] = i;
  }
  INT *d_int_a2d, *d_int_b2d, *d_int_c2d;
  cudaMalloc(&d_int_a2d, sizeof(INT) * NUM);
  cudaMalloc(&d_int_b2d, sizeof(INT) * NUM);
  cudaMalloc(&d_int_c2d, sizeof(INT) * NUM);
  cudaMemcpy(d_int_a2d, h_int_2d, sizeof(INT) * NUM, cudaMemcpyHostToDevice);
  cudaMemcpy(d_int_b2d, h_int_2d, sizeof(INT) * NUM, cudaMemcpyHostToDevice);
  cudaMemcpy(d_int_c2d, h_int_2d, sizeof(INT) * NUM, cudaMemcpyHostToDevice);
  kerneel_int<<<1, NUM>>>(d_int_a2d, d_int_b2d, d_int_c2d);
  cudaDeviceSynchronize();
}
}; // namespace nnn

int main() {

  return 0;
}