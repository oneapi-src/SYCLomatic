// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none --usm-level=none --in-root=%S --out-root=%T/out --analysis-scope-path=%S/.. %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/out/kernel_warp.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl -DBUILD_TEST  %T/out/kernel_warp.dp.cpp -o %T/out/kernel_warp.dp.o %}
// out/
// ├── kernel_warp.dp.cpp
// └── MainSourceFiles.yaml
// RUN: echo > %T/exist_check
// RUN: bash %S/../check_script.sh %T/out/kernel_warp.dp.cpp %T
// RUN: bash %S/../check_script.sh %T/out/MainSourceFiles.yaml %T
// RUN: bash %S/../check_script.sh %T/out/inc/empty.h %T
// RUN: bash %S/../check_script.sh %T/out/inc/utils.dp.hpp %T
// RUN: bash %S/../check_script.sh %T/out/src %T
// RUN: FileCheck --input-file %T/exist_check --match-full-lines %S/../ref
// RUN: rm -rf %T/out
#ifndef BUILD_TEST
#include "../inc/utils.cuh"
#include "../inc/empty.h"

//CHECK:void kernel(float *input, const sycl::nd_item<3> &item_ct1, float *smem) {
__global__ void kernel(float *input) {
  float sum = 0;
  __shared__ float smem[128];
  //CHECK:float total_sum = BlockReduceSum(sum, smem, item_ct1);
  float total_sum = BlockReduceSum(sum, smem);
}

void foo() {
  float *input = NULL;
  //CHECK:dpct::get_out_of_order_queue().submit(
  //CHECK-NEXT:  [&](sycl::handler &cgh) {
  //CHECK-NEXT:    sycl::local_accessor<float, 1> smem_acc_ct1(sycl::range<1>(128), cgh);
  //CHECK-NEXT:    dpct::access_wrapper input_acc_ct0(input, cgh);
  //CHECK-EMPTY:
  //CHECK-NEXT:    cgh.parallel_for(
  //CHECK-NEXT:      sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
  //CHECK-NEXT:      [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
  //CHECK-NEXT:        kernel(input_acc_ct0.get_raw_pointer(), item_ct1, smem_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
  //CHECK-NEXT:      });
  //CHECK-NEXT:  });
  kernel<<<1, 128>>>(input);
}

template <class ReduceOp>
//CHECK:float WarpReduce(float val, const ReduceOp &op,
//CHECK:const sycl::nd_item<3> &item_ct1) {
//CHECK-NEXT:  val = op.warp_shfl_down(val, 16, item_ct1);
__device__ float WarpReduce(float val, const ReduceOp &op) {
  val = op.warp_shfl_down(val, 16);
  return val;
}

template <size_t num>
//CHECK:void compute_mode(float *input, const sycl::nd_item<3> &item_ct1) {
__global__ void compute_mode(float *input) {
  struct MaxOp {
    //CHECK:float warp_shfl_down(float acc, int offset,
    //CHECK-NEXT: const sycl::nd_item<3> &item_ct1) const {
    //CHECK-NEXT:  return WARP_SHFL_DOWN(acc, offset, item_ct1);
    __device__ float warp_shfl_down(float acc, int offset) const {
      return WARP_SHFL_DOWN(acc, offset);
    }
  };
  float m;
  //CHECK:WarpReduce(m, MaxOp{}, item_ct1);
  WarpReduce(m, MaxOp{});
}

void foo_2(float *ptr) {
  //CHECK:dpct::get_out_of_order_queue().submit(
  //CHECK-NEXT:  [&](sycl::handler &cgh) {
  //CHECK-NEXT:    dpct::access_wrapper ptr_acc_ct0(ptr, cgh);
  //CHECK-EMPTY:
  //CHECK-NEXT:    cgh.parallel_for(
  //CHECK-NEXT:      sycl::nd_range<3>(sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
  //CHECK-NEXT:      [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
  //CHECK-NEXT:        compute_mode<8>(ptr_acc_ct0.get_raw_pointer(), item_ct1);
  //CHECK-NEXT:      });
  //CHECK-NEXT:  });
  compute_mode<8><<<1, 64>>>(ptr);
}
#endif
