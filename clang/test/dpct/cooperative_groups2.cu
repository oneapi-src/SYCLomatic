// UNSUPPORTED: cuda-8.0, cuda-9.2, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.2, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/cooperative_groups2 %s --cuda-include-path="%cuda-path/include" --use-experimental-features=nd_range_barrier,logical-group -- -x cuda --cuda-host-only -std=c++14
// RUN: FileCheck %s --match-full-lines --input-file %T/cooperative_groups2/cooperative_groups2.dp.cpp

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// X is thread_block, coalesced_group, thread_block_tile, grid_group
// sync(X), X.sync(), thread_rank(X), X.thread_rank(), X.size()

__device__ void foo() {
  // CHECK: const auto cblock = item_ct1.get_group();
  const auto cblock = cg::this_thread_block();
  // CHECK: auto block = item_ct1.get_group();
  auto block = cg::this_thread_block();

  // CHECK: const auto catile32 = item_ct1.get_sub_group();
  const auto catile32 = cg::tiled_partition<32>(block);
  // CHECK: auto atile32 = item_ct1.get_sub_group();
  auto atile32 = cg::tiled_partition<32>(block);
  // CHECK: const sycl::sub_group ctile32 = item_ct1.get_sub_group();
  const cg::thread_block_tile<32> ctile32 = cg::tiled_partition<32>(block);
  // CHECK: sycl::sub_group tile32 = item_ct1.get_sub_group();
  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(block);

  // const auto catile16 = dpct::experimental::logical_group(item_ct1, item_ct1.get_group(), 16);
  const auto catile16 = cg::tiled_partition<16>(block);
  // auto atile16 = dpct::experimental::logical_group(item_ct1, item_ct1.get_group(), 16);
  auto atile16 = cg::tiled_partition<16>(block);
  // const dpct::experimental::logical_group ctile16 = dpct::experimental::logical_group(item_ct1, item_ct1.get_group(), 16);
  const cg::thread_block_tile<16> ctile16 = cg::tiled_partition<16>(block);
  // dpct::experimental::logical_group tile16 = dpct::experimental::logical_group(item_ct1, item_ct1.get_group(), 16);
  cg::thread_block_tile<16> tile16 = cg::tiled_partition<16>(block);

  // X.meta_group_rank()
  // CHECK-COUNT-5: item_ct1.get_sub_group().get_group_linear_range();
  cg::tiled_partition<32>(block).meta_group_rank();
  catile32.meta_group_rank();
  atile32.meta_group_rank();
  ctile32.meta_group_rank();
  tile32.meta_group_rank();

  // CHECK: dpct::experimental::logical_group(item_ct1, item_ct1.get_group(), 16).get_group_linear_range();
  cg::tiled_partition<16>(block).meta_group_rank();
  // CHECK: catile16.get_group_linear_range();
  catile16.meta_group_rank();
  // CHECK: atile16.get_group_linear_range();
  atile16.meta_group_rank();
}
