// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -out-root %T/sync_api_noneusm %s --cuda-include-path="%cuda-path/include" --usm-level=none --use-experimental-features=root-group -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/sync_api_noneusm/sync_api_noneusm.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/sync_api_noneusm/sync_api_noneusm.dp.cpp -o %T/sync_api_noneusm/sync_api_noneusm.dp.o %}

// CHECK: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
#include "cooperative_groups.h"
namespace cg = cooperative_groups;
using namespace cooperative_groups;

// CHECK: #define TB(b) auto b = item_ct1.get_group();
#define TB(b) cg::thread_block b = cg::this_thread_block();

__device__ void foo(int i) {}

#define FOO(x) foo(x)

// CHECK: void k(const sycl::nd_item<3> &item_ct1) {
__global__ void k() {
  // CHECK: sycl::group<3> cta = item_ct1.get_group();
  cg::thread_block cta = cg::this_thread_block();
  // CHECK: item_ct1.barrier();
  cg::sync(cta);

  // CHECK: sycl::group<3> block = item_ct1.get_group();
  cg::thread_block block = cg::this_thread_block();
  // CHECK: item_ct1.barrier(sycl::access::fence_space::local_space);
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
  // CHECK: /*
  // CHECK-NEXT: DPCT1078:{{[0-9]+}}: Consider replacing memory_order::acq_rel with memory_order::seq_cst for correctness if strong memory order restrictions are needed.
  // CHECK-NEXT: */
  // CHECK-NEXT: sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::work_group);
  __threadfence_block();
  // CHECK: /*
  // CHECK-NEXT: DPCT1078:{{[0-9]+}}: Consider replacing memory_order::acq_rel with memory_order::seq_cst for correctness if strong memory order restrictions are needed.
  // CHECK-NEXT: */
  // CHECK-NEXT: sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
  __threadfence();
  // CHECK: /*
  // CHECK-NEXT: DPCT1078:{{[0-9]+}}: Consider replacing memory_order::acq_rel with memory_order::seq_cst for correctness if strong memory order restrictions are needed.
  // CHECK-NEXT: */
  // CHECK-NEXT: sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
  __threadfence_system();
  // CHECK: item_ct1.barrier();
  // CHECK-NEXT: sycl::all_of_group(item_ct1.get_group(), p);
  __syncthreads_and(p);
  // CHECK: item_ct1.barrier();
  // CHECK-NEXT: sycl::any_of_group(item_ct1.get_group(), p);
  __syncthreads_or(p);
  // CHECK: item_ct1.barrier();
  // CHECK-NEXT: sycl::reduce_over_group(item_ct1.get_group(), p == 0 ? 0 : 1, sycl::ext::oneapi::plus<>());
  __syncthreads_count(p);
  // CHECK: sycl::group_barrier(item_ct1.get_sub_group());
  __syncwarp(0xffffffff);

  // CHECK: int a = (item_ct1.barrier(), sycl::all_of_group(item_ct1.get_group(), p));
  int a = __syncthreads_and(p);
  // CHECK: int b = (item_ct1.barrier(), sycl::any_of_group(item_ct1.get_group(), p));
  int b = __syncthreads_or(p);
  // CHECK: int c = (item_ct1.barrier(), sycl::reduce_over_group(item_ct1.get_group(), p == 0 ? 0 : 1, sycl::ext::oneapi::plus<>()));
  int c = __syncthreads_count(p);

  // CHECK: foo((item_ct1.barrier(), sycl::all_of_group(item_ct1.get_group(), p)));
  foo(__syncthreads_and(p));
  // CHECK: foo((item_ct1.barrier(), sycl::any_of_group(item_ct1.get_group(), p)));
  foo(__syncthreads_or(p));
  // CHECK: foo((item_ct1.barrier(), sycl::reduce_over_group(item_ct1.get_group(), p == 0 ? 0 : 1, sycl::ext::oneapi::plus<>())));
  foo(__syncthreads_count(p));

  // CHECK: FOO((item_ct1.barrier(), sycl::all_of_group(item_ct1.get_group(), p)));
  FOO(__syncthreads_and(p));
  // CHECK: FOO((item_ct1.barrier(), sycl::any_of_group(item_ct1.get_group(), p)));
  FOO(__syncthreads_or(p));
  // CHECK: FOO((item_ct1.barrier(), sycl::reduce_over_group(item_ct1.get_group(), p == 0 ? 0 : 1, sycl::ext::oneapi::plus<>())));
  FOO(__syncthreads_count(p));
}

// CHECK: void kernel(const sycl::nd_item<3> &item_ct1) {
// CHECK-NEXT:   sycl::ext::oneapi::experimental::root_group grid = item_ct1.ext_oneapi_get_root_group();
// CHECK-NEXT:   sycl::group_barrier(grid);
// CHECK-NEXT: }
__global__ void kernel() {
  cg::grid_group grid = cg::this_grid();
  grid.sync();
}

int main() {
  // CHECK: {
  // CHECK-NEXT: auto exp_props = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync};
  // CHECK-EMPTY:
  // CHECK-NEXT:  dpct::get_out_of_order_queue().parallel_for(
  // CHECK-NEXT:  sycl::nd_range<3>(sycl::range<3>(1, 1, 2) * sycl::range<3>(1, 1, 2), sycl::range<3>(1, 1, 2)),
  // CHECK-NEXT:  exp_props,     [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:      kernel(item_ct1);
  // CHECK-NEXT:    });
  // CHECK-NEXT: }
  // CHECK-NEXT:  dpct::get_current_device().queues_wait_and_throw();
  kernel<<<2, 2>>>();
  cudaDeviceSynchronize();
  return 0;
}
