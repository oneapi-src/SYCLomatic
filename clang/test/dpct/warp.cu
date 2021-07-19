// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -out-root %T/warp %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/warp/warp.dp.cpp --match-full-lines %s

#include "cuda.h"

#define FULL_MASK 23
#define NUM_ELEMENTS 1024

// CHECK: void foo(sycl::nd_item<3> item_{{[0-9a-z]+}}) {
__global__ void foo() {
  unsigned mask;
  int predicate;
  int val = 0;
  int srcLane;
  unsigned delta;
  int laneMask;

  // CHECK: sycl::all_of_group(item_{{[0-9a-z]+}}.get_sub_group(), predicate);
  __all(predicate);
  //CHECK: /*
  //CHECK-NEXT: DPCT1086:{{[0-9]+}}: The function __activemask is not supported. You may need to use 0xffffffff instead or adjust the code.
  //CHECK-NEXT: */
  //CHECK-NEXT: sycl::all_of_group(item_{{[0-9a-z]+}}.get_sub_group(), (~__activemask() & (0x1 << item_{{[0-9a-z]+}}.get_sub_group().get_local_linear_id())) || predicate);
  __all_sync(__activemask(), predicate);
  // CHECK: sycl::all_of_group(item_{{[0-9a-z]+}}.get_sub_group(), (~mask & (0x1 << item_{{[0-9a-z]+}}.get_sub_group().get_local_linear_id())) || predicate);
  __all_sync(mask, predicate);

  // CHECK: sycl::any_of_group(item_{{[0-9a-z]+}}.get_sub_group(), predicate);
  __any(predicate);
  //CHECK: /*
  //CHECK-NEXT: DPCT1086:{{[0-9]+}}: The function __activemask is not supported. You may need to use 0xffffffff instead or adjust the code.
  //CHECK-NEXT: */
  //CHECK-NEXT: sycl::any_of_group(item_ct1.get_sub_group(), (__activemask() & (0x1 << item_ct1.get_sub_group().get_local_linear_id())) && predicate);
  __any_sync(__activemask(), predicate);
  // CHECK: sycl::any_of_group(item_{{[0-9a-z]+}}.get_sub_group(), (mask & (0x1 << item_{{[0-9a-z]+}}.get_sub_group().get_local_linear_id())) && predicate);
  __any_sync(mask, predicate);

  // CHECK: item_{{[0-9a-z]+}}.get_sub_group().shuffle(val, srcLane);
  __shfl(val, srcLane);
  // CHECK: /*
  // CHECK-NEXT: DPCT1023:{{[0-9]+}}: The DPC++ sub-group does not support mask options for shuffle.
  // CHECK-NEXT: */
  // CHECK-NEXT: item_{{[0-9a-z]+}}.get_sub_group().shuffle(val, srcLane);
  __shfl_sync(mask, val, srcLane);
  // CHECK: /*
  // CHECK-NEXT: DPCT1023:{{[0-9]+}}: The DPC++ sub-group does not support mask options for shuffle.
  // CHECK-NEXT: */
  // CHECK-NEXT: item_{{[0-9a-z]+}}.get_sub_group().shuffle(val, srcLane);
  __shfl_sync(mask, val, srcLane, warpSize);

  // CHECK: sycl::shift_group_right(item_{{[0-9a-z]+}}.get_sub_group(), val, delta);
  __shfl_up(val, delta);
  // CHECK: /*
  // CHECK-NEXT:DPCT1023:{{[0-9]+}}: The DPC++ sub-group does not support mask options for sycl::shift_group_right.
  // CHECK-NEXT: */
  // CHECK-NEXT: sycl::shift_group_right(item_{{[0-9a-z]+}}.get_sub_group(), val, delta);
  __shfl_up_sync(mask, val, delta);
  // CHECK: /*
  // CHECK-NEXT:DPCT1023:{{[0-9]+}}: The DPC++ sub-group does not support mask options for sycl::shift_group_right.
  // CHECK-NEXT: */
  // CHECK-NEXT: sycl::shift_group_right(item_{{[0-9a-z]+}}.get_sub_group(), val, delta);
  __shfl_up_sync(mask, val, delta, warpSize);

  // CHECK: sycl::shift_group_left(item_{{[0-9a-z]+}}.get_sub_group(), val, delta);
  __shfl_down(val, delta);
  // CHECK: /*
  // CHECK-NEXT: DPCT1023:{{[0-9]+}}: The DPC++ sub-group does not support mask options for sycl::shift_group_left.
  // CHECK-NEXT: */
  // CHECK-NEXT: sycl::shift_group_left(item_{{[0-9a-z]+}}.get_sub_group(), val, delta);
  __shfl_down_sync(mask, val, delta);
  // CHECK: /*
  // CHECK-NEXT: DPCT1023:{{[0-9]+}}: The DPC++ sub-group does not support mask options for sycl::shift_group_left.
  // CHECK-NEXT: */
  // CHECK-NEXT: sycl::shift_group_left(item_{{[0-9a-z]+}}.get_sub_group(), val, delta);
  __shfl_down_sync(mask, val, delta, warpSize);

  // CHECK: item_{{[0-9a-z]+}}.get_sub_group().shuffle_xor(val, laneMask);
  __shfl_xor(val, laneMask);
  // CHECK: /*
  // CHECK-NEXT: DPCT1023:{{[0-9]+}}: The DPC++ sub-group does not support mask options for shuffle_xor.
  // CHECK-NEXT: */
  // CHECK-NEXT: item_{{[0-9a-z]+}}.get_sub_group().shuffle_xor(val, laneMask);
  __shfl_xor_sync(mask, val, laneMask);
  // CHECK: /*
  // CHECK-NEXT: DPCT1023:{{[0-9]+}}: The DPC++ sub-group does not support mask options for shuffle_xor.
  // CHECK-NEXT: */
  // CHECK-NEXT: item_{{[0-9a-z]+}}.get_sub_group().shuffle_xor(val, laneMask);
  __shfl_xor_sync(mask, val, laneMask, warpSize);

  int input[NUM_ELEMENTS];
  // CHECK: mask = sycl::reduce_over_group(item_ct{{[0-9a-z]+}}.get_sub_group(), item_ct{{[0-9a-z]+}}.get_local_id(2) < NUM_ELEMENTS ? (0x1 << item_ct{{[0-9a-z]+}}.get_sub_group().get_local_linear_id()) : 0, sycl::ONEAPI::plus<>());
  mask = __ballot(threadIdx.x < NUM_ELEMENTS);
  // CHECK: mask = sycl::reduce_over_group(item_ct{{[0-9a-z]+}}.get_sub_group(), (__activemask() & (0x1 << item_ct{{[0-9a-z]+}}.get_sub_group().get_local_linear_id())) && item_ct{{[0-9a-z]+}}.get_local_id(2) < NUM_ELEMENTS ? (0x1 << item_ct{{[0-9a-z]+}}.get_sub_group().get_local_linear_id()) : 0, sycl::ONEAPI::plus<>());
  mask = __ballot_sync(__activemask(), threadIdx.x < NUM_ELEMENTS);
  // CHECK: mask = sycl::reduce_over_group(item_ct{{[0-9a-z]+}}.get_sub_group(), (FULL_MASK & (0x1 << item_ct{{[0-9a-z]+}}.get_sub_group().get_local_linear_id())) && item_ct{{[0-9a-z]+}}.get_local_id(2) < NUM_ELEMENTS ? (0x1 << item_ct{{[0-9a-z]+}}.get_sub_group().get_local_linear_id()) : 0, sycl::ONEAPI::plus<>());
  mask = __ballot_sync(FULL_MASK, threadIdx.x < NUM_ELEMENTS);
  //CHECK: /*
  //CHECK-NEXT: DPCT1086:{{[0-9]+}}: The function __activemask is not supported. You may need to use 0xffffffff instead or adjust the code.
  //CHECK-NEXT: */
  //CHECK-NEXT: mask = __activemask();
  mask = __activemask();
  if (threadIdx.x < NUM_ELEMENTS) {
    val = input[threadIdx.x];
    for (int offset = 16; offset > 0; offset /= 2)
      // CHECK: /*
      // CHECK-NEXT: DPCT1023:{{[0-9]+}}: The DPC++ sub-group does not support mask options for sycl::shift_group_left.
      // CHECK-NEXT: */
      // CHECK-NEXT: val += sycl::shift_group_left(item_{{[0-9a-z]+}}.get_sub_group(), val, offset);
      val += __shfl_down_sync(mask, val, offset);
  }
}

__global__ void foo2() {  
  unsigned mask;
  int predicate;
  int val = 0;
  int srcLane;
  unsigned delta;
  int laneMask;

  // CHECK: item_{{[0-9a-z]+}}.get_sub_group().shuffle(val, srcLane);
  __shfl(val, srcLane, 16);
  // CHECK: /*
  // CHECK-NEXT: DPCT1023:{{[0-9]+}}: The DPC++ sub-group does not support mask options for shuffle.
  // CHECK-NEXT: */
  // CHECK-NEXT: item_{{[0-9a-z]+}}.get_sub_group().shuffle(val, srcLane);
  __shfl_sync(mask, val, srcLane, 16);

  // CHECK: sycl::shift_group_right(item_{{[0-9a-z]+}}.get_sub_group(), val, delta);
  __shfl_up(val, delta, 16);
  // CHECK: /*
  // CHECK-NEXT:DPCT1023:{{[0-9]+}}: The DPC++ sub-group does not support mask options for sycl::shift_group_right.
  // CHECK-NEXT: */
  // CHECK-NEXT: sycl::shift_group_right(item_{{[0-9a-z]+}}.get_sub_group(), val, delta);
  __shfl_up_sync(mask, val, delta, 16);

  // CHECK: sycl::shift_group_left(item_{{[0-9a-z]+}}.get_sub_group(), val, delta);
  __shfl_down(val, delta, 16);
  // CHECK: /*
  // CHECK-NEXT: DPCT1023:{{[0-9]+}}: The DPC++ sub-group does not support mask options for sycl::shift_group_left.
  // CHECK-NEXT: */
  // CHECK-NEXT: sycl::shift_group_left(item_{{[0-9a-z]+}}.get_sub_group(), val, delta);
  __shfl_down_sync(mask, val, delta, 16);

  // CHECK: item_{{[0-9a-z]+}}.get_sub_group().shuffle_xor(val, laneMask);
  __shfl_xor(val, laneMask, 16);
  // CHECK: /*
  // CHECK-NEXT: DPCT1023:{{[0-9]+}}: The DPC++ sub-group does not support mask options for shuffle_xor.
  // CHECK-NEXT: */
  // CHECK-NEXT: item_{{[0-9a-z]+}}.get_sub_group().shuffle_xor(val, laneMask);
  __shfl_xor_sync(mask, val, laneMask, 16);
}

__global__ void foo3() {
  unsigned mask;
  int predicate;
  int val = 0;
  int srcLane;

  // CHECK: item_{{[0-9a-z]+}}.get_sub_group().shuffle(val, srcLane);
  __shfl(val, srcLane, 16);
  // CHECK: /*
  // CHECK-NEXT: DPCT1023:{{[0-9]+}}: The DPC++ sub-group does not support mask options for sycl::shift_group_left.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT:DPCT1085:{{[0-9]+}}: The function shift_group_left requires subgroup size to be 32, while other subgroup function in same DPC++ kernel requires differnt subgroup size. You may need to adjust the code.
  // CHECK-NEXT: */
  // CHECK-NEXT: sycl::shift_group_left(item_ct1.get_sub_group(), val, srcLane);
  __shfl_down_sync(mask, val, srcLane, 32);
}

int main() {
  //CHECK:   q_ct1.parallel_for(
  //CHECK-NEXT:    sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  //CHECK-NEXT:    [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
  //CHECK-NEXT:      foo(item_ct1);
  //CHECK-NEXT:    });
  foo<<<1,1>>>();

  //CHECK:   q_ct1.parallel_for(
  //CHECK-NEXT:    sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  //CHECK-NEXT:    [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(16){{\]\]}} {
  //CHECK-NEXT:      foo2(item_ct1);
  //CHECK-NEXT:    });
  foo2<<<1,1>>>();

  //CHECK:   q_ct1.parallel_for(
  //CHECK-NEXT:    sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  //CHECK-NEXT:    [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(16){{\]\]}} {
  //CHECK-NEXT:      foo3(item_ct1);
  //CHECK-NEXT:    });
  foo3<<<1,1>>>();
}