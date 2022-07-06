// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -in-root %S -out-root %T/explicit_namespace_none %S/explicit_namespace_none.cu --cuda-include-path="%cuda-path/include" --use-explicit-namespace=none --sycl-named-lambda -- -x cuda --cuda-host-only -std=c++14
// RUN: FileCheck --input-file %T/explicit_namespace_none/explicit_namespace_none.dp.cpp --match-full-lines %s

// CHECK: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: using namespace dpct;
// CHECK-NEXT: using namespace sycl;
#include <cmath>
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <cstdio>
#include <algorithm>
#include <exception>
__device__ float4 fun() {
  float4 a, b, c;
// CHECK: fma(a.x(), b.x(), c.x());
  __fmaf_rn(a.x, b.x, c.x);
// CHECK: return float4(fma(a.x(), b.x(), c.x()), fma(a.y(), b.y(), c.y()), fma(a.z(), b.z(), c.z()), fma(a.w(), b.w(), c.w()));
  return make_float4(__fmaf_rd(a.x, b.x, c.x), __fmaf_rz(a.y, b.y, c.y), __fmaf_rn(a.z, b.z, c.z), __fmaf_rn(a.w, b.w, c.w));
}


__global__ void kernel() {

}

void foo() {
// CHECK:   get_default_queue().parallel_for<dpct_kernel_name<class kernel_{{[a-f0-9]+}}>>(
// CHECK-NEXT:         nd_range<3>(range<3>(1, 1, ceil(2.3)), range<3>(1, 1, 1)),
// CHECK-NEXT:         [=](nd_item<3> item_{{[0-9a-z]+}}) {
// CHECK-NEXT:           kernel();
// CHECK-NEXT:         });
  kernel<<< ceil(2.3), 1 >>>();
}
// CHECK: global_memory<int, 0> al;
__device__ int al;
const int num_elements = 16;
// CHECK: global_memory<float, 2> fy(num_elements, 4 * num_elements);
__device__ float fx[2], fy[num_elements][4 * num_elements];
const int size = 64;
__device__ float tmp[size];
// CHECK: void kernel2(float *out, nd_item<3> [[ITEM:item_ct1]], int *al, float *fx,
// CHECK-NEXT:              dpct::accessor<float, global, 2> fy, float *tmp) {
// CHECK-NEXT:   out[{{.*}}[[ITEM]].get_local_id(2)] += *al;
// CHECK-NEXT:   fx[{{.*}}[[ITEM]].get_local_id(2)] = fy[{{.*}}[[ITEM]].get_local_id(2)][{{.*}}[[ITEM]].get_local_id(2)];
// CHECK-NEXT:   tmp[1] = 1.0f;
// CHECK-NEXT: }
__global__ void kernel2(float *out) {
  out[threadIdx.x] += al;
  fx[threadIdx.x] = fy[threadIdx.x][threadIdx.x];
  tmp[1] = 1.0f;
}

// CHECK:void test_assignment() try {
// CHECK-NEXT:  int err;
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
// CHECK-NEXT:*/
// CHECK-NEXT:  if (err = (*0 = (void *)malloc_device(0, get_default_queue()), 0)) {
// CHECK-NEXT:    printf("error!\n");
// CHECK-NEXT:  }
// CHECK-NEXT:}
// CHECK-NEXT:catch (sycl::exception const &exc) {
// CHECK-NEXT:  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
// CHECK-NEXT:  std::exit(1);
// CHECK-NEXT:}
void test_assignment() {
  cudaError_t err;
  if (err = cudaMalloc(0, 0)) {
    printf("error!\n");
  }
}

int main() {
  // CHECK: device_ext &dev_ct1 = get_current_device();
  // CHECK-NEXT: queue &q_ct1 = dev_ct1.default_queue();
  int *mapsp1D, *mapspkeyD,*mapspvalD;
  int numsH=10;

  cudaMalloc(&mapsp1D, numsH*sizeof(int));
  cudaMalloc(&mapspkeyD, numsH*sizeof(int));
  cudaMalloc(&mapspvalD, numsH*sizeof(int));

// CHECK:  device_pointer<int> mapsp1T(mapsp1D);
  thrust::device_ptr<int> mapsp1T(mapsp1D);
// CHECK:  device_pointer<int> mapspkeyT(mapspkeyD);
  thrust::device_ptr<int> mapspkeyT(mapspkeyD);
// CHECK:  device_pointer<int> mapspvalT(mapspvalD);
  thrust::device_ptr<int> mapspvalT(mapspvalD);

// CHECK:  std::copy(oneapi::dpl::execution::make_device_policy<class Policy_{{[0-9a-f]+}}>(q_ct1), mapsp1T, mapsp1T + numsH, mapspkeyT);
  thrust::copy(mapsp1T, mapsp1T + numsH, mapspkeyT);
// CHECK:  iota(oneapi::dpl::execution::make_device_policy<class Policy_{{[0-9a-f]+}}>(q_ct1), mapspvalT, mapspvalT + numsH);
  thrust::sequence(mapspvalT, mapspvalT + numsH);
// CHECK:  stable_sort(oneapi::dpl::execution::make_device_policy(q_ct1), mapspkeyT, mapspkeyT + numsH, mapspvalT);
  thrust::stable_sort_by_key(mapspkeyT, mapspkeyT + numsH, mapspvalT);
}
