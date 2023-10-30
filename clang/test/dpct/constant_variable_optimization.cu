// RUN: dpct --format-range=none --optimize-migration -out-root %T/constant_variable_optimization %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/constant_variable_optimization/constant_variable_optimization.dp.cpp

#include<cuda_runtime.h>
#include<iostream>

// CHECK: static const int a = 11;
// CHECK: static const int b[32] = {1, 2, 3};
// CHECK: static const int c = 1;
__constant__ int a = 11;
static __constant__ int b[32] = {1, 2, 3};
__constant__ int c = 1;

// CHECK: __dpct_inline__ void kernel1(int *p){
// CHECK:     *p = a;
// CHECK: }
__global__ void kernel1(int *p){
  *p = a;
}

// CHECK: __dpct_inline__ void kernel2(int *p, const sycl::nd_item<3> &item_ct1){
// CHECK:     int i = item_ct1.get_local_id(2);
// CHECK:     p[i] = b[i];
// CHECK: }
__global__ void kernel2(int *p){
    int i = threadIdx.x;
    p[i] = b[i];
}


int main(){
  int *dp;
  cudaMallocManaged(&dp, sizeof(int));
  kernel1<<<1, 1>>>(dp);
  cudaDeviceSynchronize();
  std::cout << *dp << std::endl;

  int *dp2;
  cudaMallocManaged(&dp2, 32 * sizeof(int));
  kernel2<<<1, 32>>>(dp2);
  cudaDeviceSynchronize();
  for(int i = 0; i < 32; i++) {
    std::cout << dp2[i] << std::endl;
  }
  size_t size;
// CHECK:   /*
// CHECK:   DPCT1119:{{[0-9]+}}: The migration of this CUDA Symbol APIs are not supported becuase specifier '__constant__' of symbol "c" was migrated to 'const'. You may need to adjust the code.
// CHECK:   */
// CHECK:   cudaGetSymbolSize(&size, c);
  cudaGetSymbolSize(&size, c);

  return 0;
}
