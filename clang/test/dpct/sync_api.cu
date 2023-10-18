// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -out-root %T/sync_api %s --cuda-include-path="%cuda-path/include" --use-experimental-features=nd_range_barrier,logical-group -- -x cuda --cuda-host-only -std=c++14
// RUN: FileCheck %s --match-full-lines --input-file %T/sync_api/sync_api.dp.cpp

// CHECK: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
#include "cooperative_groups.h"
namespace cg = cooperative_groups;
using namespace cooperative_groups;

// CHECK: #define TB(b) auto b = item_ct1.get_group();
#define TB(b) cg::thread_block b = cg::this_thread_block();

__device__ void foo(int i) {}

#define FOO(x) foo(x)
__device__ void test(cg::thread_block cta) {
  cg:sync(cta);
}
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

// CHECK: void kernel(const sycl::nd_item<3> &item_ct1,
// CHECK-NEXT:            sycl::atomic_ref<unsigned int, sycl::memory_order::seq_cst, sycl::memory_scope::device, sycl::access::address_space::global_space> &sync_ct1) {
// CHECK-NEXT:  dpct::experimental::nd_range_barrier(item_ct1, sync_ct1);
// CHECK-NEXT:}
__global__ void kernel() {
  cg::grid_group grid = cg::this_grid();
  grid.sync();
}

int main() {
// CHECK:  {
// CHECK-NEXT:    dpct::global_memory<unsigned int, 0> d_sync_ct1(0);
// CHECK-NEXT:    unsigned *sync_ct1 = d_sync_ct1.get_ptr(dpct::get_in_order_queue());
// CHECK-NEXT:    dpct::get_in_order_queue().memset(sync_ct1, 0, sizeof(int)).wait();
// CHECK-NEXT:    dpct::get_in_order_queue().parallel_for(
// CHECK-NEXT:      sycl::nd_range<3>(sycl::range<3>(1, 1, 2) * sycl::range<3>(1, 1, 2), sycl::range<3>(1, 1, 2)),
// CHECK-NEXT:      [=](sycl::nd_item<3> item_ct1)  {
// CHECK-NEXT:        auto atm_sync_ct1 = sycl::atomic_ref<unsigned int, sycl::memory_order::seq_cst, sycl::memory_scope::device, sycl::access::address_space::global_space>(sync_ct1[0]);
// CHECK-NEXT:        kernel(item_ct1, atm_sync_ct1);
// CHECK-NEXT:      }).wait();
// CHECK-NEXT:  }
  kernel<<<2, 2>>>();
  cudaDeviceSynchronize();
  return 0;
}

#define LOGICAL_SIZE 8

// CHECK:void foo1(sycl::group<3> &tb,
// CHECK-NEXT:   sycl::sub_group &tbt32,
// CHECK-NEXT:   const sycl::nd_item<3> &item_ct1) {
__device__ void foo1(cg::thread_block &tb,
                     cg::thread_block_tile<32> &tbt32) {
// CHECK: item_ct1.get_local_linear_id();
// CHECK-NEXT: item_ct1.get_sub_group().get_local_linear_id();
// CHECK-NEXT: item_ct1.get_local_linear_id();
// CHECK-NEXT: item_ct1.get_sub_group().get_local_linear_id();
  tb.thread_rank();
  tbt32.thread_rank();
  cg::thread_rank(tb);
  cg::thread_rank(tbt32);

// CHECK: /*
// CHECK-NEXT: DPCT1065:{{[0-9]+}}: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global memory.
// CHECK-NEXT: */
// CHECK-NEXT: item_ct1.barrier();
// CHECK-NEXT: /*
// CHECK-NEXT: DPCT1065:{{[0-9]+}}: Consider replacing sycl::sub_group::barrier() with sycl::sub_group::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global memory.
// CHECK-NEXT: */
// CHECK-NEXT: item_ct1.get_sub_group().barrier();
// CHECK-NEXT: /*
// CHECK-NEXT: DPCT1065:{{[0-9]+}}: Consider replacing sycl::nd_item::barrier() with sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global memory.
// CHECK-NEXT: */
// CHECK-NEXT: item_ct1.barrier();
// CHECK-NEXT: /*
// CHECK-NEXT: DPCT1065:{{[0-9]+}}: Consider replacing sycl::sub_group::barrier() with sycl::sub_group::barrier(sycl::access::fence_space::local_space) for better performance if there is no access to global memory.
// CHECK-NEXT: */
// CHECK-NEXT: item_ct1.get_sub_group().barrier();
  tb.sync();
  tbt32.sync();
  cg::sync(tb);
  cg::sync(tbt32);

// CHECK: dpct::experimental::logical_group tbt8 = dpct::experimental::logical_group(item_ct1, item_ct1.get_group(), LOGICAL_SIZE);
// CHECK-NEXT: int size = tbt8.get_local_linear_range();
// CHECK-NEXT: int rank = tbt8.get_local_linear_id();
// CHECK-NEXT: double temp = 0;
// CHECK-NEXT: int offset = 4;
// CHECK-NEXT: temp = dpct::shift_sub_group_left(item_ct1.get_sub_group(), temp, offset, 8);
  cg::thread_block_tile<LOGICAL_SIZE> tbt8 = cg::tiled_partition<LOGICAL_SIZE>(tb);
  int size = tbt8.size();
  int rank = tbt8.thread_rank();
  double temp = 0;
  int offset = 4;
  temp = tbt8.shfl_down(temp, offset);
}

__global__ void foo2() {
// CHECK: sycl::group<3> tb = item_ct1.get_group();
// CHECK-NEXT: sycl::sub_group tbt32 = item_ct1.get_sub_group();
  cg::thread_block tb = cg::this_thread_block();
  cg::thread_block_tile<32> tbt32 = cg::tiled_partition<32>(tb);
  foo1(tb, tbt32);
}

__global__ void foo_tile32() {
// CHECK: sycl::group<3> ttb = item_ct1.get_group();
// CHECK-NEXT: sycl::sub_group tile32 = item_ct1.get_sub_group();
// CHECK-NEXT: double rowThreadSum = 0.0;
// CHECK-NEXT: int offset= 32;
// CHECK-NEXT: sycl::shift_group_left(item_ct1.get_sub_group(), rowThreadSum, offset);
// CHECK-NEXT: item_ct1.get_sub_group().get_local_linear_id();
  cg::thread_block ttb = cg::this_thread_block();
  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(ttb);
  double rowThreadSum = 0.0;
  int offset= 32;
  tile32.shfl_down(rowThreadSum, offset);
  tile32.thread_rank();
}

int foo3() {
//CHECK: dpct::get_in_order_queue().parallel_for(
//CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
//CHECK-NEXT:   [=](sycl::nd_item<3> item_ct1) {{\[\[}}intel::reqd_sub_group_size(32){{\]\]}} {
//CHECK-NEXT:     foo2(item_ct1);
//CHECK-NEXT:   });
  foo2<<<1,1>>>();
  return 0;
}

// CHECK:      void foo4(const sycl::group<3> &crtb,
// CHECK-NEXT:           const sycl::group<3> *cptb);
__device__ void foo4(const cg::thread_block &crtb,
                     const cg::thread_block *cptb);
