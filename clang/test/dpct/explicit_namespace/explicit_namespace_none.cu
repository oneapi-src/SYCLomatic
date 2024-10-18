// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -in-root %S -out-root %T/explicit_namespace_none %S/explicit_namespace_none.cu --cuda-include-path="%cuda-path/include" --use-explicit-namespace=none --sycl-named-lambda -- -x cuda --cuda-host-only -std=c++14
// RUN: FileCheck --input-file %T/explicit_namespace_none/explicit_namespace_none.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST %T/explicit_namespace_none/explicit_namespace_none.dp.cpp -o %T/explicit_namespace_none/explicit_namespace_none.dp.o %}

// CHECK: #include <sycl/sycl.hpp>
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
#ifndef NO_BUILD_TEST
// CHECK: fma(a.x(), b.x(), c.x());
  __fmaf_rn(a.x, b.x, c.x);
// CHECK: return float4(fma(a.x(), b.x(), c.x()), fma(a.y(), b.y(), c.y()), fma(a.z(), b.z(), c.z()), fma(a.w(), b.w(), c.w()));
  return make_float4(__fmaf_rd(a.x, b.x, c.x), __fmaf_rz(a.y, b.y, c.y), __fmaf_rn(a.z, b.z, c.z), __fmaf_rn(a.w, b.w, c.w));
#endif
}


__global__ void kernel1() {

}

void foo() {
#ifndef NO_BUILD_TEST
// CHECK:   get_in_order_queue().parallel_for<dpct_kernel_name<class kernel1_{{[a-f0-9]+}}>>(
// CHECK-NEXT:         nd_range<3>(range<3>(1, 1, ceil(2.3)), range<3>(1, 1, 1)),
// CHECK-NEXT:         [=](nd_item<3> item_{{[0-9a-z]+}}) {
// CHECK-NEXT:           kernel1();
// CHECK-NEXT:         });
  kernel1<<< ceil(2.3), 1 >>>();
#endif
}
// CHECK: static global_memory<int, 0> al;
__device__ int al;
const int num_elements = 16;
// CHECK: static global_memory<float, 2> fy(num_elements, 4 * num_elements);
__device__ float fx[2], fy[num_elements][4 * num_elements];
const int size = 64;
__device__ float tmp[size];
// CHECK: void kernel2(float *out, const nd_item<3> &[[ITEM:item_ct1]], int &al, float *fx,
// CHECK-NEXT:              dpct::accessor<float, global, 2> fy, float *tmp) {
// CHECK-NEXT:   out[{{.*}}[[ITEM]].get_local_id(2)] += al;
// CHECK-NEXT:   fx[{{.*}}[[ITEM]].get_local_id(2)] = fy[{{.*}}[[ITEM]].get_local_id(2)][{{.*}}[[ITEM]].get_local_id(2)];
// CHECK-NEXT:   tmp[1] = 1.0f;
// CHECK-NEXT: }
__global__ void kernel2(float *out) {
  out[threadIdx.x] += al;
  fx[threadIdx.x] = fy[threadIdx.x][threadIdx.x];
  tmp[1] = 1.0f;
}

// CHECK:void test_assignment() try {
// CHECK-NEXT:  err0 err;
// CHECK-NEXT:  int *a;
// CHECK-NEXT:  if (err = DPCT_CHECK_ERROR(a = (int *)malloc_device(0, get_in_order_queue()))) {
// CHECK-NEXT:    printf("error!\n");
// CHECK-NEXT:  }
// CHECK-NEXT:}
// CHECK-NEXT:catch (sycl::exception const &exc) {
// CHECK-NEXT:  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
// CHECK-NEXT:  std::exit(1);
// CHECK-NEXT:}
void test_assignment() {
  cudaError_t err;
  int *a;
  if (err = cudaMalloc(&a, 0)) {
    printf("error!\n");
  }
}

int main() {
  // CHECK: device_ext &dev_ct1 = get_current_device();
  // CHECK-NEXT: queue &q_ct1 = dev_ct1.in_order_queue();
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

  // CHECK: iota(oneapi::dpl::execution::make_device_policy(q_ct1), mapspvalT, mapspvalT + numsH);
  thrust::sequence(mapspvalT, mapspvalT + numsH);
  // CHECK:  stable_sort(oneapi::dpl::execution::make_device_policy(q_ct1), mapspkeyT, mapspkeyT + numsH, mapspvalT);
  thrust::stable_sort_by_key(mapspkeyT, mapspkeyT + numsH, mapspvalT);
}
