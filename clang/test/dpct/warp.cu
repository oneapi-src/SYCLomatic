// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct -out-root %T %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/warp.dp.cpp --match-full-lines %s

#include "cuda.h"

#define FULL_MASK 23
#define NUM_ELEMENTS 1024

// CHECK: void foo(cl::sycl::nd_item<3> item_{{[0-9a-z]+}}) {
__global__ void foo() {
  unsigned mask;
  int predicate;
  int val = 0;
  int srcLane;
  unsigned delta;
  int warpSize;
  int laneMask;

  // CHECK: item_{{[0-9a-z]+}}.get_sub_group().all(predicate);
  __all(predicate);
  // CHECK: /*
  // CHECK-NEXT: DPCT1023:{{[0-9]+}}: The DPC++ sub-group does not support mask options for all.
  // CHECK-NEXT: */
  // CHECK-NEXT: item_{{[0-9a-z]+}}.get_sub_group().all(predicate);
  __all_sync(mask, predicate);

  // CHECK: item_{{[0-9a-z]+}}.get_sub_group().any(predicate);
  __any(predicate);
  // CHECK: /*
  // CHECK-NEXT: DPCT1023:{{[0-9]+}}: The DPC++ sub-group does not support mask options for any.
  // CHECK-NEXT: */
  // CHECK-NEXT: item_{{[0-9a-z]+}}.get_sub_group().any(predicate);
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

  // CHECK: item_{{[0-9a-z]+}}.get_sub_group().shuffle_up(val, delta);
  __shfl_up(val, delta);
  // CHECK: /*
  // CHECK-NEXT:DPCT1023:{{[0-9]+}}: The DPC++ sub-group does not support mask options for shuffle_up.
  // CHECK-NEXT: */
  // CHECK-NEXT: item_{{[0-9a-z]+}}.get_sub_group().shuffle_up(val, delta);
  __shfl_up_sync(mask, val, delta);
  // CHECK: /*
  // CHECK-NEXT:DPCT1023:{{[0-9]+}}: The DPC++ sub-group does not support mask options for shuffle_up.
  // CHECK-NEXT: */
  // CHECK-NEXT: item_{{[0-9a-z]+}}.get_sub_group().shuffle_up(val, delta);
  __shfl_up_sync(mask, val, delta, warpSize);

  // CHECK: item_{{[0-9a-z]+}}.get_sub_group().shuffle_down(val, delta);
  __shfl_down(val, delta);
  // CHECK: /*
  // CHECK-NEXT: DPCT1023:{{[0-9]+}}: The DPC++ sub-group does not support mask options for shuffle_down.
  // CHECK-NEXT: */
  // CHECK-NEXT: item_{{[0-9a-z]+}}.get_sub_group().shuffle_down(val, delta);
  __shfl_down_sync(mask, val, delta);
  // CHECK: /*
  // CHECK-NEXT: DPCT1023:{{[0-9]+}}: The DPC++ sub-group does not support mask options for shuffle_down.
  // CHECK-NEXT: */
  // CHECK-NEXT: item_{{[0-9a-z]+}}.get_sub_group().shuffle_down(val, delta);
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
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  // CHECK-NEXT: mask = __ballot(item_{{[0-9a-z]+}}.get_local_id(2) < NUM_ELEMENTS);
  mask = __ballot(threadIdx.x < NUM_ELEMENTS);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  // CHECK-NEXT: mask = __ballot_sync(FULL_MASK, item_{{[0-9a-z]+}}.get_local_id(2) < NUM_ELEMENTS);
  mask = __ballot_sync(FULL_MASK, threadIdx.x < NUM_ELEMENTS);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  // CHECK-NEXT: mask = __activemask();
  mask = __activemask();
  if (threadIdx.x < NUM_ELEMENTS) {
    val = input[threadIdx.x];
    for (int offset = 16; offset > 0; offset /= 2)
      // CHECK: /*
      // CHECK-NEXT: DPCT1023:{{[0-9]+}}: The DPC++ sub-group does not support mask options for shuffle_down.
      // CHECK-NEXT: */
      // CHECK-NEXT: val += item_{{[0-9a-z]+}}.get_sub_group().shuffle_down(val, offset);
      val += __shfl_down_sync(mask, val, offset);
  }
}
