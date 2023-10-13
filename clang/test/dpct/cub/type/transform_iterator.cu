// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -in-root %S -out-root %T/type/transform_iterator %S/transform_iterator.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/type/transform_iterator/transform_iterator.dp.cpp --match-full-lines %s

// CHECK:#include <oneapi/dpl/iterator>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <stdio.h>

#define N 10

// CHECK:struct UserDefMul {
// CHECK:double operator()(double d) const {
// CHECK:return d * 3.0;
// CHECK:}
// CHECK:};
struct UserDefMul {
  __device__ double operator()(double d) const {
    return d * 3.0;
  }
};


// CHECK:void compute(double *d_in, double *d_out) {
// CHECK:oneapi::dpl::transform_iterator<double *, UserDefMul> iter(d_in, UserDefMul());
// CHECK:for (int i = 0; i < N; ++i)
// CHECK:d_out[i] = iter[i];
// CHECK:}
__global__ void compute(double *d_in, double *d_out) {
  cub::TransformInputIterator<double, UserDefMul, double *> iter(d_in, UserDefMul());
  for (int i = 0; i < N; ++i)
    d_out[i] = iter[i];
}

void print_array(double *d) {
  for (int i = 0; i < N; ++i)
    printf("%.2lf ", d[i]);
  printf("\n");
}

// CHECK:void test() {
// CHECK:dpct::device_ext &dev_ct1 = dpct::get_current_device();
// CHECK:sycl::queue &q_ct1 = dev_ct1.in_order_queue();
// CHECK:double *d_in = nullptr;
// CHECK:double *d_out = nullptr;
// CHECK:double h_in[N], h_out[N];
// CHECK:d_in = sycl::malloc_device<double>(N, q_ct1);
// CHECK:d_out = sycl::malloc_device<double>(N, q_ct1);
// CHECK:for (int i = 0; i < N; ++i) h_in[i] = i;
// CHECK:q_ct1.memcpy((void *)d_in, (void *)h_in, sizeof(double) * N);
// CHECK:oneapi::dpl::transform_iterator<double *, UserDefMul> iter(d_in, UserDefMul());
// CHECK:q_ct1.parallel_for(
// CHECK:sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)), 
// CHECK:[=](sycl::nd_item<3> item_ct1) {
// CHECK:compute(d_in, d_out);
// CHECK:});
// CHECK:q_ct1.memcpy((void *)h_out, (void *)d_out, sizeof(double) * N).wait();
// CHECK:}
void test() {
  double *d_in = nullptr;
  double *d_out = nullptr;
  double h_in[N], h_out[N];
  cudaMalloc((void **)&d_in, sizeof(double) * N);
  cudaMalloc((void **)&d_out, sizeof(double) * N);
  for (int i = 0; i < N; ++i) h_in[i] = i;
  cudaMemcpy((void *)d_in, (void *)h_in, sizeof(double) * N, cudaMemcpyHostToDevice);
  cub::TransformInputIterator<double, UserDefMul, double *> iter(d_in, UserDefMul());
  compute<<<1, 1>>>(d_in, d_out);
  cudaMemcpy((void *)h_out, (void *)d_out, sizeof(double) * N, cudaMemcpyDeviceToHost);
  for (int i = 0; i < N; ++i)
    printf("%.2lf\n", h_out[i]);
}