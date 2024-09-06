// UNSUPPORTED: cuda-8.0, cuda-9.2, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.2, v10.1, v10.2
// RUN: dpct --use-syclcompat --format-range=none -out-root %T/cooperative_groups_syclcompat %s --cuda-include-path="%cuda-path/include" --use-experimental-features=nd_range_barrier,logical-group -- -x cuda --cuda-host-only -std=c++14
// RUN: FileCheck %s --match-full-lines --input-file %T/cooperative_groups_syclcompat/cooperative_groups_syclcompat.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/cooperative_groups_syclcompat/cooperative_groups_syclcompat.dp.cpp -o %T/cooperative_groups_syclcompat/cooperative_groups_syclcompat.dp.o %}

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// X is thread_block, coalesced_group, thread_block_tile, grid_group
// sync(X), X.sync(), thread_rank(X), X.thread_rank(), X.size()

__device__ void foo() {
  // CHECK: auto block = item_ct1.get_group();
  auto block = cg::this_thread_block();

  // CHECK: auto group_x = syclcompat::dim3(block.get_group_id(2), block.get_group_id(1), block.get_group_id(0)).x;
  // CHECK-NEXT: auto thread_x = syclcompat::dim3(block.get_local_id(2), block.get_local_id(1), block.get_local_id(0)).x;
  auto group_x = block.group_index().x;
  auto thread_x = block.thread_index().x;

  const cg::thread_block_tile<32> ctile32 = cg::tiled_partition<32>(block);
  // CHECK: sycl::sub_group tile32 = item_ct1.get_sub_group();
  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(block);

  // CHECK: const syclcompat::experimental::logical_group ctile16 = syclcompat::experimental::logical_group(item_ct1, item_ct1.get_group(), 16);
  const cg::thread_block_tile<16> ctile16 = cg::tiled_partition<16>(block);
  // CHECK: syclcompat::experimental::logical_group tile16 = syclcompat::experimental::logical_group(item_ct1, item_ct1.get_group(), 16);
  cg::thread_block_tile<16> tile16 = cg::tiled_partition<16>(block);
}
