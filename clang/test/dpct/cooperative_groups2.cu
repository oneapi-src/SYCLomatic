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

  // CHECK: auto group_x = item_ct1.get_global_id()[2];
  // CHECK-NEXT: auto thread_x = item_ct1.get_local_id()[2];
  auto group_x = block.group_index().x;
  auto thread_x = block.thread_index().x;

  // CHECK: const auto catile32 = item_ct1.get_sub_group();
  const auto catile32 = cg::tiled_partition<32>(block);
  // CHECK: auto atile32 = item_ct1.get_sub_group();
  auto atile32 = cg::tiled_partition<32>(block);
  // CHECK: const sycl::sub_group ctile32 = item_ct1.get_sub_group();
  const cg::thread_block_tile<32> ctile32 = cg::tiled_partition<32>(block);
  // CHECK: sycl::sub_group tile32 = item_ct1.get_sub_group();
  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(block);

  // X.meta_group_rank()
  // CHECK-COUNT-5: item_ct1.get_sub_group().get_group_linear_id();
  cg::tiled_partition<32>(block).meta_group_rank();
  catile32.meta_group_rank();
  atile32.meta_group_rank();
  ctile32.meta_group_rank();
  tile32.meta_group_rank();

  // CHECK: const auto catile16 = dpct::experimental::logical_group(item_ct1, item_ct1.get_group(), 16);
  const auto catile16 = cg::tiled_partition<16>(block);
  // CHECK: auto atile16 = dpct::experimental::logical_group(item_ct1, item_ct1.get_group(), 16);
  auto atile16 = cg::tiled_partition<16>(block);
  // CHECK: const dpct::experimental::logical_group ctile16 = dpct::experimental::logical_group(item_ct1, item_ct1.get_group(), 16);
  const cg::thread_block_tile<16> ctile16 = cg::tiled_partition<16>(block);
  // CHECK: dpct::experimental::logical_group tile16 = dpct::experimental::logical_group(item_ct1, item_ct1.get_group(), 16);
  cg::thread_block_tile<16> tile16 = cg::tiled_partition<16>(block);
  // CHECK: dpct::experimental::logical_group(item_ct1, item_ct1.get_group(), 16).get_group_linear_id();
  cg::tiled_partition<16>(block).meta_group_rank();
  // CHECK: catile16.get_group_linear_id();
  catile16.meta_group_rank();
  // CHECK: atile16.get_group_linear_id();
  atile16.meta_group_rank();

  // CHECK: const auto catile8 = dpct::experimental::logical_group(item_ct1, item_ct1.get_group(), 8);
  const auto catile8 = cg::tiled_partition<8>(block);
  // CHECK: auto atile8 = dpct::experimental::logical_group(item_ct1, item_ct1.get_group(), 8);
  auto atile8 = cg::tiled_partition<8>(block);
  // CHECK: const dpct::experimental::logical_group ctile8 = dpct::experimental::logical_group(item_ct1, item_ct1.get_group(), 8);
  const cg::thread_block_tile<8> ctile8 = cg::tiled_partition<8>(block);
  // CHECK: dpct::experimental::logical_group tile8 = dpct::experimental::logical_group(item_ct1, item_ct1.get_group(), 8);
  cg::thread_block_tile<8> tile8 = cg::tiled_partition<8>(block);
  // CHECK: dpct::experimental::logical_group(item_ct1, item_ct1.get_group(), 8).get_group_linear_id();
  cg::tiled_partition<8>(block).meta_group_rank();
  // CHECK: catile8.get_group_linear_id();
  catile8.meta_group_rank();
  // CHECK: atile8.get_group_linear_id();
  atile8.meta_group_rank();

  // CHECK: const auto catile4 = dpct::experimental::logical_group(item_ct1, item_ct1.get_group(), 4);
  const auto catile4 = cg::tiled_partition<4>(block);
  // CHECK: auto atile4 = dpct::experimental::logical_group(item_ct1, item_ct1.get_group(), 4);
  auto atile4 = cg::tiled_partition<4>(block);
  // CHECK: const dpct::experimental::logical_group ctile4 = dpct::experimental::logical_group(item_ct1, item_ct1.get_group(), 4);
  const cg::thread_block_tile<4> ctile4 = cg::tiled_partition<4>(block);
  // CHECK: dpct::experimental::logical_group tile4 = dpct::experimental::logical_group(item_ct1, item_ct1.get_group(), 4);
  cg::thread_block_tile<4> tile4 = cg::tiled_partition<4>(block);
  // CHECK: dpct::experimental::logical_group(item_ct1, item_ct1.get_group(), 4).get_group_linear_id();
  cg::tiled_partition<4>(block).meta_group_rank();
  // CHECK: catile4.get_group_linear_id();
  catile4.meta_group_rank();
  // CHECK: atile4.get_group_linear_id();
  atile4.meta_group_rank();

  // CHECK: const auto catile2 = dpct::experimental::logical_group(item_ct1, item_ct1.get_group(), 2);
  const auto catile2 = cg::tiled_partition<2>(block);
  // CHECK: auto atile2 = dpct::experimental::logical_group(item_ct1, item_ct1.get_group(), 2);
  auto atile2 = cg::tiled_partition<2>(block);
  // CHECK: const dpct::experimental::logical_group ctile2 = dpct::experimental::logical_group(item_ct1, item_ct1.get_group(), 2);
  const cg::thread_block_tile<2> ctile2 = cg::tiled_partition<2>(block);
  // CHECK: dpct::experimental::logical_group tile2 = dpct::experimental::logical_group(item_ct1, item_ct1.get_group(), 2);
  cg::thread_block_tile<2> tile2 = cg::tiled_partition<2>(block);
  // CHECK: dpct::experimental::logical_group(item_ct1, item_ct1.get_group(), 2).get_group_linear_id();
  cg::tiled_partition<2>(block).meta_group_rank();
  // CHECK: catile2.get_group_linear_id();
  catile2.meta_group_rank();
  // CHECK: atile2.get_group_linear_id();
  atile2.meta_group_rank();

  // CHECK: const auto catile1 = dpct::experimental::logical_group(item_ct1, item_ct1.get_group(), 1);
  const auto catile1 = cg::tiled_partition<1>(block);
  // CHECK: auto atile1 = dpct::experimental::logical_group(item_ct1, item_ct1.get_group(), 1);
  auto atile1 = cg::tiled_partition<1>(block);
  // CHECK: const dpct::experimental::logical_group ctile1 = dpct::experimental::logical_group(item_ct1, item_ct1.get_group(), 1);
  const cg::thread_block_tile<1> ctile1 = cg::tiled_partition<1>(block);
  // CHECK: dpct::experimental::logical_group tile1 = dpct::experimental::logical_group(item_ct1, item_ct1.get_group(), 1);
  cg::thread_block_tile<1> tile1 = cg::tiled_partition<1>(block);
  // CHECK: dpct::experimental::logical_group(item_ct1, item_ct1.get_group(), 1).get_group_linear_id();
  cg::tiled_partition<1>(block).meta_group_rank();
  // CHECK: catile1.get_group_linear_id();
  catile1.meta_group_rank();
  // CHECK: atile1.get_group_linear_id();
  atile1.meta_group_rank();
}

__global__ void test_const_ref() {
    const cooperative_groups::thread_block &cta = cooperative_groups::this_thread_block();
    const cooperative_groups::thread_block *cta4 = &cta;
}
