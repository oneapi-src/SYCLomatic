// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --optimize-migration --format-range=none -out-root %T/warp_2 %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/warp_2/warp_2.dp.cpp --match-full-lines %s

#include "cuda.h"

#define FULL_MASK 23
#define NUM_ELEMENTS 1024


__global__ void kernel1() {
  int val;
  int srcLane;
  // CHECK: sycl::select_from_group(item_{{[0-9a-z]+}}.get_sub_group(), val, srcLane);
  __shfl(val, srcLane);
}

__global__ void kernel2() {
  unsigned mask;
  int val;
  int srcLane;
  // CHECK: /*
  // CHECK-NEXT: DPCT1023:{{[0-9]+}}: The DPC++ sub-group does not support mask options for sycl::select_from_group.
  // CHECK-NEXT: */
  // CHECK-NEXT: sycl::select_from_group(item_{{[0-9a-z]+}}.get_sub_group(), val, srcLane);
  __shfl_sync(mask, val, srcLane);
}

__global__ void kernel3() {
  unsigned delta;
  int val;
  // CHECK: sycl::shift_group_right(item_{{[0-9a-z]+}}.get_sub_group(), val, delta);
  __shfl_up(val, delta);
}

__global__ void kernel4() {
  unsigned mask;
  int val;
  unsigned delta;
  // CHECK: /*
  // CHECK-NEXT:DPCT1023:{{[0-9]+}}: The DPC++ sub-group does not support mask options for sycl::shift_group_right.
  // CHECK-NEXT: */
  // CHECK-NEXT: sycl::shift_group_right(item_{{[0-9a-z]+}}.get_sub_group(), val, delta);
  __shfl_up_sync(mask, val, delta);
}

__global__ void kernel5() {
  int val;
  unsigned delta;
  // CHECK: sycl::shift_group_left(item_{{[0-9a-z]+}}.get_sub_group(), val, delta);
  __shfl_down(val, delta);
}

__global__ void kernel6() {
  unsigned mask;
  int val;
  unsigned delta;
  // CHECK: /*
  // CHECK-NEXT: DPCT1023:{{[0-9]+}}: The DPC++ sub-group does not support mask options for sycl::shift_group_left.
  // CHECK-NEXT: */
  // CHECK-NEXT: sycl::shift_group_left(item_{{[0-9a-z]+}}.get_sub_group(), val, delta);
  __shfl_down_sync(mask, val, delta);
}

__global__ void kernel7() {
  int laneMask;
  int val;
  // CHECK: sycl::permute_group_by_xor(item_{{[0-9a-z]+}}.get_sub_group(), val, laneMask);
  __shfl_xor(val, laneMask);
}

__global__ void kernel8() {
  unsigned mask;
  int val;
  int laneMask;
  // CHECK: /*
  // CHECK-NEXT: DPCT1023:{{[0-9]+}}: The DPC++ sub-group does not support mask options for sycl::permute_group_by_xor.
  // CHECK-NEXT: */
  // CHECK-NEXT: sycl::permute_group_by_xor(item_{{[0-9a-z]+}}.get_sub_group(), val, laneMask);
  __shfl_xor_sync(mask, val, laneMask);
}


int main() {

  // CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
  // CHECK: sycl::queue &q_ct1 = dev_ct1.default_queue();


  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_{{[0-9a-z]+}}) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
  // CHECK-NEXT:     kernel1(item_{{[0-9a-z]+}});
  // CHECK-NEXT:   });
  kernel1<<<1,1>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_{{[0-9a-z]+}}) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
  // CHECK-NEXT:     kernel2(item_{{[0-9a-z]+}});
  // CHECK-NEXT:   });
  kernel2<<<1,1>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_{{[0-9a-z]+}}) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
  // CHECK-NEXT:     kernel3(item_{{[0-9a-z]+}});
  // CHECK-NEXT:   });
  kernel3<<<1,1>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_{{[0-9a-z]+}}) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
  // CHECK-NEXT:     kernel4(item_{{[0-9a-z]+}});
  // CHECK-NEXT:   });
  kernel4<<<1,1>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_{{[0-9a-z]+}}) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
  // CHECK-NEXT:     kernel5(item_{{[0-9a-z]+}});
  // CHECK-NEXT:   });
  kernel5<<<1,1>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_{{[0-9a-z]+}}) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
  // CHECK-NEXT:     kernel6(item_{{[0-9a-z]+}});
  // CHECK-NEXT:   });
  kernel6<<<1,1>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_{{[0-9a-z]+}}) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
  // CHECK-NEXT:     kernel7(item_{{[0-9a-z]+}});
  // CHECK-NEXT:   });
  kernel7<<<1,1>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_{{[0-9a-z]+}}) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
  // CHECK-NEXT:     kernel8(item_{{[0-9a-z]+}});
  // CHECK-NEXT:   });
  kernel8<<<1,1>>>();
}