// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct  --format-range=none --assume-nd-range-dim=1  -out-root %T/kernel_1d_range_sync_api %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/kernel_1d_range_sync_api/kernel_1d_range_sync_api.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c  -DNO_BUILD_TEST -fsycl %T/kernel_1d_range_sync_api/kernel_1d_range_sync_api.dp.cpp -o %T/kernel_1d_range_sync_api/kernel_1d_range_sync_api.dp.o %}


#include "cooperative_groups.h"
namespace cg = cooperative_groups;
using namespace cooperative_groups;

// CHECK: void global1() {
__global__ void global1() {
  // CHECK: sycl::group<1> cta = sycl::ext::oneapi::this_work_item::get_work_group<1>();
  // CHECK: const sycl::group<1> &cta2 = sycl::ext::oneapi::this_work_item::get_work_group<1>();
  cg::thread_block cta = cg::this_thread_block();
  const cg::thread_block &cta2 = cg::this_thread_block();

  // CHECK: sycl::group<1> block = sycl::ext::oneapi::this_work_item::get_work_group<1>();
  cg::thread_block block = cg::this_thread_block();

  // CHECK: sycl::group<1> b0 = sycl::ext::oneapi::this_work_item::get_work_group<1>(), b1 = sycl::ext::oneapi::this_work_item::get_work_group<1>();
  cg::thread_block b0 = cg::this_thread_block(), b1 = cg::this_thread_block();
  thread_block b111 = cg::this_thread_block();

  // CHECK: int GroupSize = sycl::ext::oneapi::this_work_item::get_work_group<1>().get_local_linear_range();
  int GroupSize = block.size();
}
#ifndef NO_BUILD_TEST
// CHECK: #define TB(b) auto b = sycl::ext::oneapi::this_work_item::get_work_group<dpct_placeholder /* Fix the dimension manually */>();
#define TB(b) cg::thread_block b = cg::this_thread_block();

// CHECK: void global2() {
__global__ void global2() {
  TB(blk);
}

// CHECK: void global3() {
__global__ void global3() {
  TB(blk);
}

int foo5() {
  //CHECK:q_ct1.parallel_for(
  //CHECK-NEXT:      sycl::nd_range<1>(sycl::range<1>(1), sycl::range<1>(1)),
  //CHECK-NEXT:      [=](sycl::nd_item<1> item_ct1) {
  //CHECK-NEXT:        global1();
  //CHECK-NEXT:      });
  global1<<<1,1>>>();

  //CHECK:q_ct1.parallel_for(
  //CHECK-NEXT:      sycl::nd_range<1>(sycl::range<1>(1), sycl::range<1>(1)),
  //CHECK-NEXT:      [=](sycl::nd_item<1> item_ct1) {
  //CHECK-NEXT:        global2();
  //CHECK-NEXT:      });
  global2<<<1,1>>>();

  //CHECK:q_ct1.parallel_for(
  //CHECK-NEXT:      sycl::nd_range<3>(sycl::range<3>(3, 2, 1), sycl::range<3>(1, 1, 1)),
  //CHECK-NEXT:      [=](sycl::nd_item<3> item_ct1) {
  //CHECK-NEXT:        global3();
  //CHECK-NEXT:      });
  global3<<<dim3(1,2,3),1>>>();

  return 0;
}
#endif