// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/sync_api.dp.cpp

// CHECK: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
#include "cooperative_groups.h"
namespace cg = cooperative_groups;
using namespace cooperative_groups;

// CHECK: #define TB(b) sycl::group<3> b = item_ct1.get_group();
#define TB(b) cg::thread_block b = cg::this_thread_block();

__device__ void foo(int i) {}

#define FOO(x) foo(x)

// CHECK: void k(sycl::nd_item<3> item_ct1) {
__global__ void k() {
  // CHECK: sycl::group<3> cta = item_ct1.get_group();
  cg::thread_block cta = cg::this_thread_block();
  // CHECK: item_ct1.barrier();
  cg::sync(cta);

  // CHECK: sycl::group<3> block = item_ct1.get_group();
  cg::thread_block block = cg::this_thread_block();
  // CHECK: item_ct1.barrier();
  __syncthreads();
  // CHECK: item_ct1.barrier();
  block.sync();
  // CHECK: item_ct1.barrier();
  cg::sync(block);
  // CHECK: item_ct1.barrier();
  cg::this_thread_block().sync();
  // CHECK: item_ct1.barrier();
  cg::sync(cg::this_thread_block());

  // CHECK: sycl::group<3> b0 = item_ct1.get_group(), b1 = item_ct1.get_group();
  cg::thread_block b0 = cg::this_thread_block(), b1 = cg::this_thread_block();

  TB(blk);

  int p;
  // CHECK: item_ct1.mem_fence();
  __threadfence_block();
  // CHECK: item_ct1.barrier();
  // CHECK-NEXT: sycl::ONEAPI::all_of(item_ct1.get_group(), p);
  __syncthreads_and(p);
  // CHECK: item_ct1.barrier();
  // CHECK-NEXT: sycl::ONEAPI::any_of(item_ct1.get_group(), p);
  __syncthreads_or(p);

  // CHECK: int a = (item_ct1.barrier(), sycl::ONEAPI::all_of(item_ct1.get_group(), p));
  int a = __syncthreads_and(p);
  // CHECK: int b = (item_ct1.barrier(), sycl::ONEAPI::any_of(item_ct1.get_group(), p));
  int b = __syncthreads_or(p);

  // CHECK: foo((item_ct1.barrier(), sycl::ONEAPI::all_of(item_ct1.get_group(), p)));
  foo(__syncthreads_and(p));
  // CHECK: foo((item_ct1.barrier(), sycl::ONEAPI::any_of(item_ct1.get_group(), p)));
  foo(__syncthreads_or(p));

  // CHECK: FOO((item_ct1.barrier(), sycl::ONEAPI::all_of(item_ct1.get_group(), p)));
  FOO(__syncthreads_and(p));
  // CHECK: FOO((item_ct1.barrier(), sycl::ONEAPI::any_of(item_ct1.get_group(), p)));
  FOO(__syncthreads_or(p));
}
