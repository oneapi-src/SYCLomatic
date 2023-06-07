// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -out-root %T/warp_mask %s --use-experimental-features=masked-sub-group-operation --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/warp_mask/warp_mask.dp.cpp --match-full-lines %s

#include "cuda.h"

#define FULL_MASK 23
#define NUM_ELEMENTS 1024


__global__ void kernel1() {
  unsigned mask;
  int val;
  int srcLane;
  // CHECK: /*
  // CHECK: DPCT1108:{{[0-9]+}}: '__shfl_sync' was migrated with the experimental feature masked sub_group function which may not be supported by all compilers or runtimes. You may need to adjust the code.
  // CHECK: */
  // CHECK: dpct::experimental::select_from_sub_group(mask, item_{{[0-9a-z]+}}.get_sub_group(), val, srcLane);
  __shfl_sync(mask, val, srcLane);
}

__global__ void kernel2() {
  unsigned mask;
  int val;
  unsigned delta;
  // CHECK: /*
  // CHECK: DPCT1108:{{[0-9]+}}: '__shfl_up_sync' was migrated with the experimental feature masked sub_group function which may not be supported by all compilers or runtimes. You may need to adjust the code.
  // CHECK: */
  // CHECK: dpct::experimental::shift_sub_group_right(mask, item_{{[0-9a-z]+}}.get_sub_group(), val, delta);
  __shfl_up_sync(mask, val, delta);
}

__global__ void kernel3() {
  unsigned mask;
  int val;
  unsigned delta;
  // CHECK: /*
  // CHECK: DPCT1108:{{[0-9]+}}: '__shfl_down_sync' was migrated with the experimental feature masked sub_group function which may not be supported by all compilers or runtimes. You may need to adjust the code.
  // CHECK: */
  // CHECK: dpct::experimental::shift_sub_group_left(mask, item_{{[0-9a-z]+}}.get_sub_group(), val, delta);
  __shfl_down_sync(mask, val, delta);
}

__global__ void kernel4() {
  unsigned mask;
  int val;
  int laneMask;
  // CHECK: /*
  // CHECK: DPCT1108:{{[0-9]+}}: '__shfl_xor_sync' was migrated with the experimental feature masked sub_group function which may not be supported by all compilers or runtimes. You may need to adjust the code.
  // CHECK: */
  // CHECK: dpct::experimental::permute_sub_group_by_xor(mask, item_{{[0-9a-z]+}}.get_sub_group(), val, laneMask);
  __shfl_xor_sync(mask, val, laneMask);
}

int main() {

  // CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
  // CHECK: sycl::queue &q_ct1 = dev_ct1.default_queue();

  auto BS = dim3(1);
  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(BS, BS),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_{{[0-9a-z]+}}) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
  // CHECK-NEXT:     kernel1(item_{{[0-9a-z]+}});
  // CHECK-NEXT:   });
  kernel1<<<1,BS>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_{{[0-9a-z]+}}) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
  // CHECK-NEXT:     kernel2(item_{{[0-9a-z]+}});
  // CHECK-NEXT:   });
  kernel2<<<1,32>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_{{[0-9a-z]+}}) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
  // CHECK-NEXT:     kernel3(item_{{[0-9a-z]+}});
  // CHECK-NEXT:   });
  kernel3<<<1,32>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_{{[0-9a-z]+}}) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
  // CHECK-NEXT:     kernel4(item_{{[0-9a-z]+}});
  // CHECK-NEXT:   });
  kernel4<<<1,32>>>();
}