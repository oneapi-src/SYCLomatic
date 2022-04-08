// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: c2s --format-range=none -out-root %T/thrust-header %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17
// RUN: FileCheck --input-file %T/thrust-header/thrust-header.dp.cpp --match-full-lines %s
// CHECK: #include <oneapi/dpl/execution>
// CHECK-NEXT: #include <oneapi/dpl/algorithm>
// CHECK-NEXT: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <c2s/c2s.hpp>
// CHECK-NEXT: #include <cstdio>
// CHECK-NEXT: #include <algorithm>
#include <cstdio>
#include <algorithm>
// CHECK: #include <c2s/dpl_utils.hpp>
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
int main() {
  // CHECK: c2s::device_ext &dev_ct1 = c2s::get_current_device();
  // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.default_queue();
  int *mapsp1D, *mapspkeyD,*mapspvalD;
  int numsH=10;

  cudaMalloc(&mapsp1D, numsH*sizeof(int));
  cudaMalloc(&mapspkeyD, numsH*sizeof(int));
  cudaMalloc(&mapspvalD, numsH*sizeof(int));

// CHECK:  c2s::device_pointer<int> mapsp1T(mapsp1D);
  thrust::device_ptr<int> mapsp1T(mapsp1D);
// CHECK:  c2s::device_pointer<int> mapspkeyT(mapspkeyD);
  thrust::device_ptr<int> mapspkeyT(mapspkeyD);
// CHECK:  c2s::device_pointer<int> mapspvalT(mapspvalD);
  thrust::device_ptr<int> mapspvalT(mapspvalD);

// CHECK:  std::copy(oneapi::dpl::execution::make_device_policy(q_ct1), mapsp1T, mapsp1T + numsH, mapspkeyT);
  thrust::copy(mapsp1T, mapsp1T + numsH, mapspkeyT);
// CHECK:  c2s::iota(oneapi::dpl::execution::make_device_policy(q_ct1), mapspvalT, mapspvalT + numsH);
  thrust::sequence(mapspvalT, mapspvalT + numsH);
// CHECK:  c2s::stable_sort(oneapi::dpl::execution::make_device_policy(q_ct1), mapspkeyT, mapspkeyT + numsH, mapspvalT);
  thrust::stable_sort_by_key(mapspkeyT, mapspkeyT + numsH, mapspvalT);
}

