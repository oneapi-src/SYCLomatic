// RUN: dpct --format-range=none --optimize-migration -out-root %T/constant_variable_optimization %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/constant_variable_optimization/constant_variable_optimization.dp.cpp

#include<cuda_runtime.h>
#include<iostream>


// CHECK: static const int dev_a = 11;
// CHECK: static const int dev_b = {1, 2, 3};
// CHECK: static dpct::constant_memory<int, 0> dev_c(1);
// CHECK: static dpct::constant_memory<int, 1> dev_d(sycl::range<1>(5), {22});
__constant__ int dev_a = 11;
static __constant__ int dev_b[32] = {1, 2, 3};
__constant__ int dev_c = 1;
__constant__ int dev_d[5] = {22};



// CHECK: __dpct_inline__ void kernel1(int *ptr, int dev_c){
// CHECK:     *ptr = dev_a;
// CHECK:     *ptr = dev_c;
// CHECK: }
__global__ void kernel1(int *ptr){
  *ptr = dev_a;
  *ptr = dev_c;
}


// CHECK: __dpct_inline__ void kernel2(int *ptr, const sycl::nd_item<3> &item_ct1, int const *dev_d){
// CHECK:     int i = item_ct1.get_local_id(2);
// CHECK:     ptr[i] = dev_b[i];
// CHECK:     ptr[i] = dev_d[i];
// CHECK: }
__global__ void kernel2(int *ptr){
    int i = threadIdx.x;
    ptr[i] = dev_b[i];
    ptr[i] = dev_d[i];
}


int main(){
  int *dp;
  cudaMallocManaged(&dp, sizeof(int));
// CHECK:   q_ct1.submit(
// CHECK:     [&](sycl::handler &cgh) {
// CHECK:       dev_c.init();
// CHECK:       auto dev_c_ptr_ct1 = dev_c.get_ptr();
// CHECK:       cgh.parallel_for(
// CHECK:         sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
// CHECK:         [=](sycl::nd_item<3> item_ct1) {
// CHECK:           kernel1(dp, *dev_c_ptr_ct1);
// CHECK:         });
// CHECK:     });
  kernel1<<<1, 1>>>(dp);
  cudaDeviceSynchronize();
  std::cout << *dp << std::endl;

  int *dp2;
  cudaMallocManaged(&dp2, 32 * sizeof(int));
// CHECK:   q_ct1.submit(
// CHECK:     [&](sycl::handler &cgh) {
// CHECK:       dev_d.init();
// CHECK:       auto dev_d_ptr_ct1 = dev_d.get_ptr();
// CHECK:       cgh.parallel_for(
// CHECK:         sycl::nd_range<3>(sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
// CHECK:         [=](sycl::nd_item<3> item_ct1) {
// CHECK:           kernel2(dp2, item_ct1, dev_d_ptr_ct1);
// CHECK:         });
// CHECK:     });
  kernel2<<<1, 32>>>(dp2);
  cudaDeviceSynchronize();
  for(int i = 0; i < 32; i++) {
    std::cout << dp2[i] << std::endl;
  }
  size_t size;
// CHECK:   size = dev_c.get_size();
// CHECK:   size = dev_d.get_size();
  cudaGetSymbolSize(&size, dev_c);
  cudaGetSymbolSize(&size, dev_d);
  return 0;
}