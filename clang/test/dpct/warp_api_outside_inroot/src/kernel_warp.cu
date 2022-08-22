// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none --usm-level=none --in-root=%S --out-root=%T/out --analysis-scope-path=%S/.. %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
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
#include "../inc/utils.cuh"
#include "../inc/empty.h"

//CHECK:void kernel(float *input, sycl::nd_item<3> item_ct1, float *smem) {
__global__ void kernel(float *input) {
  float sum = 0;
  __shared__ float smem[128];
  //CHECK:float total_sum = BlockReduceSum(sum, smem, item_ct1);
  float total_sum = BlockReduceSum(sum, smem);
}

void foo() {
  float *input = NULL;
  //CHECK:dpct::get_default_queue().submit(
  //CHECK-NEXT:  [&](sycl::handler &cgh) {
  //CHECK-NEXT:    sycl::accessor<float, 1, sycl::access_mode::read_write, sycl::access::target::local> smem_acc_ct1(sycl::range<1>(128), cgh);
  //CHECK-NEXT:    dpct::access_wrapper<float *> input_acc_ct0(input, cgh);
  //CHECK-EMPTY:
  //CHECK-NEXT:    cgh.parallel_for(
  //CHECK-NEXT:      sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)), 
  //CHECK-NEXT:      [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
  //CHECK-NEXT:        kernel(input_acc_ct0.get_raw_pointer(), item_ct1, smem_acc_ct1.get_pointer());
  //CHECK-NEXT:      });
  //CHECK-NEXT:  });
  kernel<<<1, 128>>>(input);
}