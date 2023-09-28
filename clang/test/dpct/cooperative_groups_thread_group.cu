// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0
// RUN: dpct --format-range=none  -use-experimental-features=free-function-queries,logical-group -out-root %T/cooperative_groups_thread_group %s --cuda-include-path="%cuda-path/include" --extra-arg="-std=c++14"
// RUN: FileCheck %s --match-full-lines --input-file %T/cooperative_groups_thread_group/cooperative_groups_thread_group.dp.cpp

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#define test33(a) testThreadGroup(a)
#define test22(a) test33(a)
#define test11(a) test22(a)
#define test44(a) testThreadGroup(a)

namespace cg = cooperative_groups;

// CHECK:void testThreadGroup(dpct::experimental::group_base<3> g) {
__device__ void testThreadGroup(cg::thread_group g) {
  // CHECK:  g.get_local_linear_id();
  g.thread_rank();
  // CHECK:  g.barrier();
  g.sync();
  // CHECK:  g.get_local_linear_range();
  g.size();

  auto block = cg::this_thread_block();
  // CHECK: item_ct1.get_local_id();
  block.thread_index();
}

__global__ void kernelFunc() {
  auto block = cg::this_thread_block();
  // CHECK: item_ct1.get_local_id();
  block.thread_index();
  // CHECK:  auto threadBlockGroup = sycl::ext::oneapi::experimental::this_group<3>();
  auto threadBlockGroup = cg::this_thread_block();


  // CHECK:  testThreadGroup(dpct::experimental::group(threadBlockGroup, item_ct1));
  testThreadGroup(threadBlockGroup);
  // CHECK:  dpct::experimental::logical_group tilePartition16 = dpct::experimental::logical_group(item_ct1, sycl::ext::oneapi::experimental::this_group<3>(), 16);
  cg::thread_block_tile<16> tilePartition16 = cg::tiled_partition<16>(threadBlockGroup);
  // CHECK:  testThreadGroup(dpct::experimental::group(tilePartition16, item_ct1));
  testThreadGroup(tilePartition16);
  // CHECK:  sycl::sub_group tilePartition32 = sycl::ext::oneapi::experimental::this_sub_group();
  cg::thread_block_tile<32> tilePartition32 = cg::tiled_partition<32>(threadBlockGroup);
  // CHECK:  testThreadGroup(dpct::experimental::group(tilePartition32, item_ct1));
  testThreadGroup(tilePartition32);
  // CHECK:  dpct::experimental::logical_group tilePartition16_1(dpct::experimental::logical_group(item_ct1, sycl::ext::oneapi::experimental::this_group<3>(), 16));
  // CHECK:  sycl::sub_group tilePartition32_2(sycl::ext::oneapi::experimental::this_sub_group());
  cg::thread_block_tile<16> tilePartition16_1(cg::tiled_partition<16>(threadBlockGroup));
  cg::thread_block_tile<32> tilePartition32_2(cg::tiled_partition<32>(threadBlockGroup));

  test11(tilePartition16);
  testThreadGroup(tilePartition16);
  test44(tilePartition16);
  test11(tilePartition32);
  test11(threadBlockGroup);
}


int main() {
  kernelFunc<<<1,1>>>();
  return 0;
}
