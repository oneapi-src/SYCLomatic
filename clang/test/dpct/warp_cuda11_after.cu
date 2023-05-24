// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct -out-root %T/warp_cuda11_after %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/warp_cuda11_after/warp_cuda11_after.dp.cpp --match-full-lines %s

#include "cuda.h"
#include "cuda_runtime.h"

__global__ void reduce_add_sync() {
  unsigned mask;
  int val;
  // CHECK: sycl::reduce_over_group(item_ct1.get_sub_group(), val, sycl::plus<>());
  __reduce_add_sync(mask, val);
}

__global__ void reduce_min_sync() {
  unsigned mask;
  int val;
  // CHECK: sycl::reduce_over_group(item_ct1.get_sub_group(), val, sycl::minimum());
  __reduce_min_sync(mask, val);
}

__global__ void reduce_max_sync() {
  unsigned mask;
  int val;
  // CHECK: sycl::reduce_over_group(item_ct1.get_sub_group(), val, sycl::maximum());
  __reduce_max_sync(mask, val);
}

__global__ void reduce_and_sync() {
  unsigned mask;
  int val;
  // CHECK: sycl::reduce_over_group(item_ct1.get_sub_group(), val, sycl::bit_and<>());
  __reduce_and_sync(mask, val);
}

__global__ void reduce_or_sync() {
  unsigned mask;
  int val;
  // CHECK: sycl::reduce_over_group(item_ct1.get_sub_group(), val, sycl::bit_or<>());
  __reduce_or_sync(mask, val);
}

__global__ void reduce_xor_sync() {
  unsigned mask;
  int val;
  // CHECK: sycl::reduce_over_group(item_ct1.get_sub_group(), val, sycl::bit_xor<>());
  __reduce_xor_sync(mask, val);
}

int main() {
  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_{{[0-9a-z]+}}) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
  // CHECK-NEXT:     reduce_add_sync(item_{{[0-9a-z]+}});
  // CHECK-NEXT:   });
  reduce_add_sync<<<1, 32>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_{{[0-9a-z]+}}) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
  // CHECK-NEXT:     reduce_min_sync(item_{{[0-9a-z]+}});
  // CHECK-NEXT:   });
  reduce_min_sync<<<1, 32>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_{{[0-9a-z]+}}) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
  // CHECK-NEXT:     reduce_max_sync(item_{{[0-9a-z]+}});
  // CHECK-NEXT:   });
  reduce_max_sync<<<1, 32>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_{{[0-9a-z]+}}) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
  // CHECK-NEXT:     reduce_and_sync(item_{{[0-9a-z]+}});
  // CHECK-NEXT:   });
  reduce_and_sync<<<1, 32>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_{{[0-9a-z]+}}) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
  // CHECK-NEXT:     reduce_or_sync(item_{{[0-9a-z]+}});
  // CHECK-NEXT:   });
  reduce_or_sync<<<1, 32>>>();

  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_{{[0-9a-z]+}}) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
  // CHECK-NEXT:     reduce_xor_sync(item_{{[0-9a-z]+}});
  // CHECK-NEXT:   });
  reduce_xor_sync<<<1, 32>>>();

  return 0;
}
