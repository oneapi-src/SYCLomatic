// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -out-root %T/warp %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/warp/warp.dp.cpp --match-full-lines %s

#include "cuda.h"

#define FULL_MASK 23
#define NUM_ELEMENTS 1024

__global__ void kernel1() {
  int predicate;
  // CHECK: sycl::all_of_group(item_{{[0-9a-z]+}}.get_sub_group(), predicate);
  __all(predicate);
}

__global__ void kernel2() {
  int predicate;
  //CHECK: /*
  //CHECK-NEXT: DPCT1086:{{[0-9]+}}: Migration of __activemask is not supported. You may need to use 0xffffffff instead or adjust the code.
  //CHECK-NEXT: */
  //CHECK-NEXT: sycl::all_of_group(item_{{[0-9a-z]+}}.get_sub_group(), (~__activemask() & (0x1 << item_{{[0-9a-z]+}}.get_sub_group().get_local_linear_id())) || predicate);
  __all_sync(__activemask(), predicate);
}

__global__ void kernel3() {
  int predicate;
  unsigned mask;
  // CHECK: sycl::all_of_group(item_{{[0-9a-z]+}}.get_sub_group(), (~mask & (0x1 << item_{{[0-9a-z]+}}.get_sub_group().get_local_linear_id())) || predicate);
  __all_sync(mask, predicate);
}

__global__ void kernel4() {
  int predicate;
  // CHECK: sycl::any_of_group(item_{{[0-9a-z]+}}.get_sub_group(), predicate);
  __any(predicate);
}

__global__ void kernel5() {
  int predicate;
  //CHECK: /*
  //CHECK-NEXT: DPCT1086:{{[0-9]+}}: Migration of __activemask is not supported. You may need to use 0xffffffff instead or adjust the code.
  //CHECK-NEXT: */
  //CHECK-NEXT: sycl::any_of_group(item_ct1.get_sub_group(), (__activemask() & (0x1 << item_ct1.get_sub_group().get_local_linear_id())) && predicate);
  __any_sync(__activemask(), predicate);
}

__global__ void kernel6() {
  int predicate;
  unsigned mask;
  // CHECK: sycl::any_of_group(item_{{[0-9a-z]+}}.get_sub_group(), (mask & (0x1 << item_{{[0-9a-z]+}}.get_sub_group().get_local_linear_id())) && predicate);
  __any_sync(mask, predicate);
}

__global__ void kernel7() {
  int val;
  int srcLane;
  // CHECK: item_{{[0-9a-z]+}}.get_sub_group().shuffle(val, srcLane);
  __shfl(val, srcLane);
}

__global__ void kernel8() {
  unsigned mask;
  int val;
  int srcLane;
  // CHECK: /*
  // CHECK-NEXT: DPCT1023:{{[0-9]+}}: The DPC++ sub-group does not support mask options for shuffle.
  // CHECK-NEXT: */
  // CHECK-NEXT: item_{{[0-9a-z]+}}.get_sub_group().shuffle(val, srcLane);
  __shfl_sync(mask, val, srcLane);
}

__global__ void kernel9() {
  unsigned mask;
  int val;
  int srcLane;
  // CHECK: /*
  // CHECK-NEXT: DPCT1023:{{[0-9]+}}: The DPC++ sub-group does not support mask options for shuffle.
  // CHECK-NEXT: */
  // CHECK-NEXT: item_{{[0-9a-z]+}}.get_sub_group().shuffle(val, srcLane);
  __shfl_sync(mask, val, srcLane, warpSize);
}

__global__ void kernel10() {
  unsigned delta;
  int val;
  // CHECK: sycl::shift_group_right(item_{{[0-9a-z]+}}.get_sub_group(), val, delta);
  __shfl_up(val, delta);
}

__global__ void kernel11() {
  unsigned mask;
  int val;
  unsigned delta;
  // CHECK: /*
  // CHECK-NEXT:DPCT1023:{{[0-9]+}}: The DPC++ sub-group does not support mask options for sycl::shift_group_right.
  // CHECK-NEXT: */
  // CHECK-NEXT: sycl::shift_group_right(item_{{[0-9a-z]+}}.get_sub_group(), val, delta);
  __shfl_up_sync(mask, val, delta);
}

__global__ void kernel12() {
  unsigned mask;
  int val;
  unsigned delta;
  // CHECK: /*
  // CHECK-NEXT:DPCT1023:{{[0-9]+}}: The DPC++ sub-group does not support mask options for sycl::shift_group_right.
  // CHECK-NEXT: */
  // CHECK-NEXT: sycl::shift_group_right(item_{{[0-9a-z]+}}.get_sub_group(), val, delta);
  __shfl_up_sync(mask, val, delta, warpSize);
}

__global__ void kernel13() {
  int val;
  unsigned delta;
  // CHECK: sycl::shift_group_left(item_{{[0-9a-z]+}}.get_sub_group(), val, delta);
  __shfl_down(val, delta);
}

__global__ void kernel14() {
  unsigned mask;
  int val;
  unsigned delta;
  // CHECK: /*
  // CHECK-NEXT: DPCT1023:{{[0-9]+}}: The DPC++ sub-group does not support mask options for sycl::shift_group_left.
  // CHECK-NEXT: */
  // CHECK-NEXT: sycl::shift_group_left(item_{{[0-9a-z]+}}.get_sub_group(), val, delta);
  __shfl_down_sync(mask, val, delta);
}

__global__ void kernel15() {
  unsigned mask;
  int val;
  unsigned delta;
  // CHECK: /*
  // CHECK-NEXT: DPCT1023:{{[0-9]+}}: The DPC++ sub-group does not support mask options for sycl::shift_group_left.
  // CHECK-NEXT: */
  // CHECK-NEXT: sycl::shift_group_left(item_{{[0-9a-z]+}}.get_sub_group(), val, delta);
  __shfl_down_sync(mask, val, delta, warpSize);
}

__global__ void kernel16() {
  int laneMask;
  int val;
  // CHECK: item_{{[0-9a-z]+}}.get_sub_group().shuffle_xor(val, laneMask);
  __shfl_xor(val, laneMask);
}

__global__ void kernel17() {
  unsigned mask;
  int val;
  int laneMask;
  // CHECK: /*
  // CHECK-NEXT: DPCT1023:{{[0-9]+}}: The DPC++ sub-group does not support mask options for shuffle_xor.
  // CHECK-NEXT: */
  // CHECK-NEXT: item_{{[0-9a-z]+}}.get_sub_group().shuffle_xor(val, laneMask);
  __shfl_xor_sync(mask, val, laneMask);
}

__global__ void kernel18() {
  unsigned mask;
  int val;
  int laneMask;
  // CHECK: /*
  // CHECK-NEXT: DPCT1023:{{[0-9]+}}: The DPC++ sub-group does not support mask options for shuffle_xor.
  // CHECK-NEXT: */
  // CHECK-NEXT: item_{{[0-9a-z]+}}.get_sub_group().shuffle_xor(val, laneMask);
  __shfl_xor_sync(mask, val, laneMask, warpSize);
}

__global__ void kernel19() {
  unsigned mask;
  // CHECK: mask = sycl::reduce_over_group(item_ct{{[0-9a-z]+}}.get_sub_group(), item_ct{{[0-9a-z]+}}.get_local_id(2) < NUM_ELEMENTS ? (0x1 << item_ct{{[0-9a-z]+}}.get_sub_group().get_local_linear_id()) : 0, sycl::ext::oneapi::plus<>());
  mask = __ballot(threadIdx.x < NUM_ELEMENTS);
}

__global__ void kernel20() {
  unsigned mask;
  // CHECK: mask = sycl::reduce_over_group(item_ct{{[0-9a-z]+}}.get_sub_group(), (__activemask() & (0x1 << item_ct{{[0-9a-z]+}}.get_sub_group().get_local_linear_id())) && item_ct{{[0-9a-z]+}}.get_local_id(2) < NUM_ELEMENTS ? (0x1 << item_ct{{[0-9a-z]+}}.get_sub_group().get_local_linear_id()) : 0, sycl::ext::oneapi::plus<>());
  mask = __ballot_sync(__activemask(), threadIdx.x < NUM_ELEMENTS);
}

__global__ void kernel21() {
  unsigned mask;
  // CHECK: mask = sycl::reduce_over_group(item_ct{{[0-9a-z]+}}.get_sub_group(), (FULL_MASK & (0x1 << item_ct{{[0-9a-z]+}}.get_sub_group().get_local_linear_id())) && item_ct{{[0-9a-z]+}}.get_local_id(2) < NUM_ELEMENTS ? (0x1 << item_ct{{[0-9a-z]+}}.get_sub_group().get_local_linear_id()) : 0, sycl::ext::oneapi::plus<>());
  mask = __ballot_sync(FULL_MASK, threadIdx.x < NUM_ELEMENTS);
}

__global__ void kernel22() {
  unsigned mask;
  //CHECK: /*
  //CHECK-NEXT: DPCT1086:{{[0-9]+}}: Migration of __activemask is not supported. You may need to use 0xffffffff instead or adjust the code.
  //CHECK-NEXT: */
  //CHECK-NEXT: mask = __activemask();
  mask = __activemask();
}

__global__ void kernel23() {
  int val;
  int srcLane;
  // CHECK: item_{{[0-9a-z]+}}.get_sub_group().shuffle(val, srcLane);
  __shfl(val, srcLane, 16);
}

__global__ void kernel24() {
  unsigned mask;
  int val;
  int srcLane;
  // CHECK: /*
  // CHECK-NEXT: DPCT1023:{{[0-9]+}}: The DPC++ sub-group does not support mask options for shuffle.
  // CHECK-NEXT: */
  // CHECK-NEXT: item_{{[0-9a-z]+}}.get_sub_group().shuffle(val, srcLane);
  __shfl_sync(mask, val, srcLane, 16);
}

__global__ void kernel25() {
  int val;
  unsigned delta;
  // CHECK: sycl::shift_group_right(item_{{[0-9a-z]+}}.get_sub_group(), val, delta);
  __shfl_up(val, delta, 16);
}

__global__ void kernel26() {
  unsigned mask;
  int val;
  unsigned delta;
  // CHECK: /*
  // CHECK-NEXT:DPCT1023:{{[0-9]+}}: The DPC++ sub-group does not support mask options for sycl::shift_group_right.
  // CHECK-NEXT: */
  // CHECK-NEXT: sycl::shift_group_right(item_{{[0-9a-z]+}}.get_sub_group(), val, delta);
  __shfl_up_sync(mask, val, delta, 16);
}

__global__ void kernel27() {
  int val;
  unsigned delta;
  // CHECK: sycl::shift_group_left(item_{{[0-9a-z]+}}.get_sub_group(), val, delta);
  __shfl_down(val, delta, 16);
}

__global__ void kernel28() {
  unsigned mask;
  int val;
  unsigned delta;
  // CHECK: /*
  // CHECK-NEXT: DPCT1023:{{[0-9]+}}: The DPC++ sub-group does not support mask options for sycl::shift_group_left.
  // CHECK-NEXT: */
  // CHECK-NEXT: sycl::shift_group_left(item_{{[0-9a-z]+}}.get_sub_group(), val, delta);
  __shfl_down_sync(mask, val, delta, 16);
}

__global__ void kernel29() {
  int val;
  int laneMask;
  // CHECK: item_{{[0-9a-z]+}}.get_sub_group().shuffle_xor(val, laneMask);
  __shfl_xor(val, laneMask, 16);
}

__global__ void kernel30() {
  unsigned mask;
  int val;
  int laneMask;
  // CHECK: /*
  // CHECK-NEXT: DPCT1023:{{[0-9]+}}: The DPC++ sub-group does not support mask options for shuffle_xor.
  // CHECK-NEXT: */
  // CHECK-NEXT: item_{{[0-9a-z]+}}.get_sub_group().shuffle_xor(val, laneMask);
  __shfl_xor_sync(mask, val, laneMask, 16);
}

__global__ void kernel31() {
  unsigned mask;
  int val;
  int srcLane;
  // CHECK: item_{{[0-9a-z]+}}.get_sub_group().shuffle(val, srcLane);
  __shfl(val, srcLane, 16);
  // CHECK: /*
  // CHECK-NEXT: DPCT1023:{{[0-9]+}}: The DPC++ sub-group does not support mask options for sycl::shift_group_left.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT:DPCT1085:{{[0-9]+}}: The function shift_group_left requires subgroup size to be 32, while other subgroup function in same DPC++ kernel requires different subgroup size. You may need to adjust the code.
  // CHECK-NEXT: */
  // CHECK-NEXT: sycl::shift_group_left(item_ct1.get_sub_group(), val, srcLane);
  __shfl_down_sync(mask, val, srcLane, 32);
}

__global__ void kernel32() {
  unsigned mask;
  int val;
  int laneMask;
  int warpSize;
  // CHECK: /*
  // CHECK-NEXT: DPCT1023:{{[0-9]+}}: The DPC++ sub-group does not support mask options for shuffle_xor.
  // CHECK-NEXT: */
  // CHECK-NEXT: item_{{[0-9a-z]+}}.get_sub_group().shuffle_xor(val, laneMask);
  __shfl_xor_sync(mask, val, laneMask, warpSize);
}

__global__ void kernel33() {
  unsigned mask;
  int val;
  int laneMask;
  int WS;
  // CHECK: /*
  // CHECK-NEXT: DPCT1023:{{[0-9]+}}: The DPC++ sub-group does not support mask options for shuffle_xor.
  // CHECK-NEXT: */
  // CHECK-NEXT: item_{{[0-9a-z]+}}.get_sub_group().shuffle_xor(val, laneMask);
  __shfl_xor_sync(mask, val, laneMask, WS);
}

int main() {

  // CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
  // CHECK: sycl::queue &q_ct1 = dev_ct1.default_queue();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
  // CHECK-NEXT:     kernel1(item_ct1);
  // CHECK-NEXT:   });
  kernel1<<<1,1>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
  // CHECK-NEXT:     kernel2(item_ct1);
  // CHECK-NEXT:   });
  kernel2<<<1,1>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
  // CHECK-NEXT:     kernel3(item_ct1);
  // CHECK-NEXT:   });
  kernel3<<<1,1>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
  // CHECK-NEXT:     kernel4(item_ct1);
  // CHECK-NEXT:   });
  kernel4<<<1,1>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
  // CHECK-NEXT:     kernel5(item_ct1);
  // CHECK-NEXT:   });
  kernel5<<<1,1>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
  // CHECK-NEXT:     kernel6(item_ct1);
  // CHECK-NEXT:   });
  kernel6<<<1,1>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
  // CHECK-NEXT:     kernel7(item_ct1);
  // CHECK-NEXT:   });
  kernel7<<<1,1>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
  // CHECK-NEXT:     kernel8(item_ct1);
  // CHECK-NEXT:   });
  kernel8<<<1,1>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
  // CHECK-NEXT:     kernel9(item_ct1);
  // CHECK-NEXT:   });
  kernel9<<<1,1>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
  // CHECK-NEXT:     kernel10(item_ct1);
  // CHECK-NEXT:   });
  kernel10<<<1,1>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
  // CHECK-NEXT:     kernel11(item_ct1);
  // CHECK-NEXT:   });
  kernel11<<<1,1>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
  // CHECK-NEXT:     kernel12(item_ct1);
  // CHECK-NEXT:   });
  kernel12<<<1,1>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
  // CHECK-NEXT:     kernel13(item_ct1);
  // CHECK-NEXT:   });
  kernel13<<<1,1>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
  // CHECK-NEXT:     kernel14(item_ct1);
  // CHECK-NEXT:   });
  kernel14<<<1,1>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
  // CHECK-NEXT:     kernel15(item_ct1);
  // CHECK-NEXT:   });
  kernel15<<<1,1>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
  // CHECK-NEXT:     kernel16(item_ct1);
  // CHECK-NEXT:   });
  kernel16<<<1,1>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
  // CHECK-NEXT:     kernel17(item_ct1);
  // CHECK-NEXT:   });
  kernel17<<<1,1>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
  // CHECK-NEXT:     kernel18(item_ct1);
  // CHECK-NEXT:   });
  kernel18<<<1,1>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
  // CHECK-NEXT:     kernel19(item_ct1);
  // CHECK-NEXT:   });
  kernel19<<<1,1>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
  // CHECK-NEXT:     kernel20(item_ct1);
  // CHECK-NEXT:   });
  kernel20<<<1,1>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
  // CHECK-NEXT:     kernel21(item_ct1);
  // CHECK-NEXT:   });
  kernel21<<<1,1>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:     kernel22();
  // CHECK-NEXT:   });
  kernel22<<<1,1>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(16){{\]\]}} {
  // CHECK-NEXT:     kernel23(item_ct1);
  // CHECK-NEXT:   });
  kernel23<<<1,1>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(16){{\]\]}} {
  // CHECK-NEXT:     kernel24(item_ct1);
  // CHECK-NEXT:   });
  kernel24<<<1,1>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(16){{\]\]}} {
  // CHECK-NEXT:     kernel25(item_ct1);
  // CHECK-NEXT:   });
  kernel25<<<1,1>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(16){{\]\]}} {
  // CHECK-NEXT:     kernel26(item_ct1);
  // CHECK-NEXT:   });
  kernel26<<<1,1>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(16){{\]\]}} {
  // CHECK-NEXT:     kernel27(item_ct1);
  // CHECK-NEXT:   });
  kernel27<<<1,1>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(16){{\]\]}} {
  // CHECK-NEXT:     kernel28(item_ct1);
  // CHECK-NEXT:   });
  kernel28<<<1,1>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(16){{\]\]}} {
  // CHECK-NEXT:     kernel29(item_ct1);
  // CHECK-NEXT:   });
  kernel29<<<1,1>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(16){{\]\]}} {
  // CHECK-NEXT:     kernel30(item_ct1);
  // CHECK-NEXT:   });
  kernel30<<<1,1>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(16){{\]\]}} {
  // CHECK-NEXT:     kernel31(item_ct1);
  // CHECK-NEXT:   });
  kernel31<<<1,1>>>();

  // CHECK: /*
  // CHECK-NEXT: DPCT1089:{{[0-9]+}}: The value of the subgroup size attribute argument 'warpSize' cannot be evaluated by the Intel(R) DPC++ Compatibility Tool. Replace "dpct_placeholder" with integral constant expression.
  // CHECK-NEXT: */
  // CHECK-NEXT: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(dpct_placeholder){{\]\]}} {
  // CHECK-NEXT:     kernel32(item_ct1);
  // CHECK-NEXT:   });
  kernel32<<<1,1>>>();

  // CHECK: /*
  // CHECK-NEXT: DPCT1089:{{[0-9]+}}: The value of the subgroup size attribute argument 'WS' cannot be evaluated by the Intel(R) DPC++ Compatibility Tool. Replace "dpct_placeholder" with integral constant expression.
  // CHECK-NEXT: */
  // CHECK-NEXT: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(dpct_placeholder){{\]\]}} {
  // CHECK-NEXT:     kernel33(item_ct1);
  // CHECK-NEXT:   });
  kernel33<<<1,1>>>();
}