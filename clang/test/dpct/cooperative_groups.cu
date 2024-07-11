// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -out-root %T/cooperative_groups %s --cuda-include-path="%cuda-path/include" --use-experimental-features=nd_range_barrier,logical-group -- -x cuda --cuda-host-only -std=c++14
// RUN: FileCheck %s --match-full-lines --input-file %T/cooperative_groups/cooperative_groups.dp.cpp

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// X is thread_block, coalesced_group, thread_block_tile, grid_group
// sync(X), X.sync(), thread_rank(X), X.thread_rank(), X.size()

__device__ void foo() {
  // CHECK: const auto cblock = item_ct1.get_group();
  const auto cblock = cg::this_thread_block();
  // CHECK: auto block = item_ct1.get_group();
  auto block = cg::this_thread_block();

  // thread_block_group is a template of two arguments,
  // first argument is size, second is what type it was created from.
  // the second argument has a default value of void, so
  // the type of cg::tiled_partition<32>(block)
  // is not cg::thread_block_tile<32>, it type is roughly like
  // cg::thread_block_tile<32, decltype(block)>.
  
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

  const auto cgg = cg::this_grid();
  auto gg = cg::this_grid();
  cg::grid_group gg2 = cg::this_grid();

  const auto cct = cg::coalesced_threads();
  auto ct = cg::coalesced_threads();

  // sync(X)
  // CHECK-COUNT-5: item_ct1.barrier();
  cg::sync(cg::this_thread_block());
  cg::sync(block);
  cg::sync(cblock);
  cg::sync(*&block);
  cg::sync(*&cblock);

  // CHECK-COUNT-2: item_ct1.barrier();
  cg::sync(ct);
  cg::sync(cct);

  // CHECK-COUNT-5: item_ct1.get_sub_group().barrier();
  cg::sync(cg::tiled_partition<32>(block));
  cg::sync(catile32);
  cg::sync(atile32);
  cg::sync(ctile32);
  cg::sync(tile32);

  // CHECK-COUNT-3: dpct::experimental::nd_range_barrier(item_ct1, sync_ct1);
  cg::sync(cg::this_grid());
  cg::sync(cgg);
  cg::sync(gg);

  // thread_rank(X)
  // CHECK-COUNT-5: item_ct1.get_local_linear_id();
  cg::thread_rank(cg::this_thread_block());
  cg::thread_rank(block);
  cg::thread_rank(cblock);
  cg::thread_rank(*&block);
  cg::thread_rank(*&cblock);

  // CHECK-COUNT-5: item_ct1.get_sub_group().get_local_linear_id();
  cg::thread_rank(cg::tiled_partition<32>(block));
  cg::thread_rank(catile32);
  cg::thread_rank(atile32);
  cg::thread_rank(ctile32);
  cg::thread_rank(tile32);

  // CHECK: dpct::experimental::logical_group(item_ct1, item_ct1.get_group(), 16).get_local_linear_id();
  cg::thread_rank(cg::tiled_partition<16>(block));
  // CHECK: catile16.get_local_linear_id();
  cg::thread_rank(catile16);
  // CHECK: atile16.get_local_linear_id();
  cg::thread_rank(atile16);
  // CHECK: ctile16.get_local_linear_id();
  cg::thread_rank(ctile16);
  // CHECK: tile16.get_local_linear_id();
  cg::thread_rank(tile16);

  // X.sync()  
  // CHECK-COUNT-4: item_ct1.barrier();
  cg::this_thread_block().sync();
  block.sync();
  cblock.sync();
  cct.sync();
  ct.sync();  

  // CHECK-COUNT-4: item_ct1.barrier();
  (&block)->sync();
  (&cblock)->sync();
  (&cct)->sync();
  (&ct)->sync();

  // CHECK-COUNT-5: item_ct1.get_sub_group().barrier();
  cg::tiled_partition<32>(block).sync();
  catile32.sync();
  atile32.sync();
  ctile32.sync();
  tile32.sync();

  // CHECK-COUNT-3: dpct::experimental::nd_range_barrier(item_ct1, sync_ct1);
  cg::this_grid().sync();
  cgg.sync();
  gg.sync();

  // X.thread_rank()
  // CHECK-COUNT-5: item_ct1.get_local_linear_id();
  cg::this_thread_block().thread_rank();
  block.thread_rank();
  cblock.thread_rank();
  (*&block).thread_rank();
  (*&cblock).thread_rank();

  // CHECK-COUNT-5: item_ct1.get_sub_group().get_local_linear_id();
  cg::tiled_partition<32>(block).thread_rank();
  catile32.thread_rank();
  atile32.thread_rank();
  ctile32.thread_rank();
  tile32.thread_rank();

  // CHECK: dpct::experimental::logical_group(item_ct1, item_ct1.get_group(), 16).get_local_linear_id();
  cg::tiled_partition<16>(block).thread_rank();
  // CHECK: catile16.get_local_linear_id();
  catile16.thread_rank();
  // CHECK: atile16.get_local_linear_id();
  atile16.thread_rank();
  // CHECK: ctile16.get_local_linear_id();
  ctile16.thread_rank();
  // CHECK: tile16.get_local_linear_id();
  tile16.thread_rank();

  // X.size()
  // CHECK-COUNT-5: item_ct1.get_group().get_local_linear_range();
  cg::this_thread_block().size();
  block.size();
  cblock.size();
  (*&block).size();
  (*&cblock).size();

  // CHECK-COUNT-5: item_ct1.get_sub_group().get_local_linear_range();
  cg::tiled_partition<32>(block).size();
  catile32.size();
  atile32.size();
  ctile32.size();
  tile32.size();

  // CHECK: dpct::experimental::logical_group(item_ct1, item_ct1.get_group(), 16).get_local_linear_range();
  cg::tiled_partition<16>(block).size();
  // CHECK: catile16.get_local_linear_range();
  catile16.size();
  // CHECK: atile16.get_local_linear_range();
  atile16.size();
  // CHECK: ctile16.get_local_linear_range();
  ctile16.size();
  // CHECK: tile16.get_local_linear_range();
  tile16.size();

  // X.shfl_down()
  // CHECK-COUNT-5: sycl::shift_group_left(item_ct1.get_sub_group(), 1, 0);
  cg::tiled_partition<32>(block).shfl_down(1, 0);
  catile32.shfl_down(1, 0);
  atile32.shfl_down(1, 0);
  ctile32.shfl_down(1, 0);
  tile32.shfl_down(1, 0);

  // CHECK-COUNT-5: dpct::shift_sub_group_left(item_ct1.get_sub_group(), 1, 0, 16);
  cg::tiled_partition<16>(block).shfl_down(1, 0);
  catile16.shfl_down(1, 0);
  atile16.shfl_down(1, 0);
  ctile16.shfl_down(1, 0);
  tile16.shfl_down(1, 0);

  // CHECK: dpct::shift_sub_group_left(item_ct1.get_sub_group(), 1, 0, 8);
  cg::tiled_partition<8>(block).shfl_down(1, 0);
  // CHECK: dpct::shift_sub_group_left(item_ct1.get_sub_group(), 1, 0, 4);
  cg::tiled_partition<4>(block).shfl_down(1, 0);
  // CHECK: dpct::shift_sub_group_left(item_ct1.get_sub_group(), 1, 0, 2);
  cg::tiled_partition<2>(block).shfl_down(1, 0);
  // CHECK: dpct::shift_sub_group_left(item_ct1.get_sub_group(), 1, 0, 1);
  cg::tiled_partition<1>(block).shfl_down(1, 0);

  // X.shfl()
  // CHECK-COUNT-5: sycl::select_from_group(item_ct1.get_sub_group(), 1, 0);
  cg::tiled_partition<32>(block).shfl(1, 0);
  catile32.shfl(1, 0);
  atile32.shfl(1, 0);
  ctile32.shfl(1, 0);
  tile32.shfl(1, 0);

  // CHECK-COUNT-5: dpct::select_from_sub_group(item_ct1.get_sub_group(), 1, 0, 16);
  cg::tiled_partition<16>(block).shfl(1, 0);
  catile16.shfl(1, 0);
  atile16.shfl(1, 0);
  ctile16.shfl(1, 0);
  tile16.shfl(1, 0);

  // CHECK: dpct::select_from_sub_group(item_ct1.get_sub_group(), 1, 0, 8);
  cg::tiled_partition<8>(block).shfl(1, 0);
  // CHECK: dpct::select_from_sub_group(item_ct1.get_sub_group(), 1, 0, 4);
  cg::tiled_partition<4>(block).shfl(1, 0);
  // CHECK: dpct::select_from_sub_group(item_ct1.get_sub_group(), 1, 0, 2);
  cg::tiled_partition<2>(block).shfl(1, 0);
  // CHECK: dpct::select_from_sub_group(item_ct1.get_sub_group(), 1, 0, 1);
  cg::tiled_partition<1>(block).shfl(1, 0);

  // X.shfl_up()
  // CHECK-COUNT-5: sycl::shift_group_right(item_ct1.get_sub_group(), 1, 0);
  cg::tiled_partition<32>(block).shfl_up(1, 0);
  catile32.shfl_up(1, 0);
  atile32.shfl_up(1, 0);
  ctile32.shfl_up(1, 0);
  tile32.shfl_up(1, 0);

  // CHECK-COUNT-5: dpct::shift_sub_group_right(item_ct1.get_sub_group(), 1, 0, 16);
  cg::tiled_partition<16>(block).shfl_up(1, 0);
  catile16.shfl_up(1, 0);
  atile16.shfl_up(1, 0);
  ctile16.shfl_up(1, 0);
  tile16.shfl_up(1, 0);

  // CHECK: dpct::shift_sub_group_right(item_ct1.get_sub_group(), 1, 0, 8);
  cg::tiled_partition<8>(block).shfl_up(1, 0);
  // CHECK: dpct::shift_sub_group_right(item_ct1.get_sub_group(), 1, 0, 4);
  cg::tiled_partition<4>(block).shfl_up(1, 0);
  // CHECK: dpct::shift_sub_group_right(item_ct1.get_sub_group(), 1, 0, 2);
  cg::tiled_partition<2>(block).shfl_up(1, 0);
  // CHECK: dpct::shift_sub_group_right(item_ct1.get_sub_group(), 1, 0, 1);
  cg::tiled_partition<1>(block).shfl_up(1, 0);

  // X.shfl_xor()
  // CHECK-COUNT-5: sycl::permute_group_by_xor(item_ct1.get_sub_group(), 1, 0);
  cg::tiled_partition<32>(block).shfl_xor(1, 0);
  catile32.shfl_xor(1, 0);
  atile32.shfl_xor(1, 0);
  ctile32.shfl_xor(1, 0);
  tile32.shfl_xor(1, 0);

  // CHECK-COUNT-5: dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), 1, 0, 16);
  cg::tiled_partition<16>(block).shfl_xor(1, 0);
  catile16.shfl_xor(1, 0);
  atile16.shfl_xor(1, 0);
  ctile16.shfl_xor(1, 0);
  tile16.shfl_xor(1, 0);

  // CHECK: dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), 1, 0, 8);
  cg::tiled_partition<8>(block).shfl_xor(1, 0);
  // CHECK: dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), 1, 0, 4);
  cg::tiled_partition<4>(block).shfl_xor(1, 0);
  // CHECK: dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), 1, 0, 2);
  cg::tiled_partition<2>(block).shfl_xor(1, 0);
  // CHECK: dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), 1, 0, 1);
  cg::tiled_partition<1>(block).shfl_xor(1, 0);
}
