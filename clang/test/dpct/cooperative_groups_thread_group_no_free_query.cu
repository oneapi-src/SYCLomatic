// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0
// RUN: dpct --format-range=none  -use-experimental-features=logical-group -out-root %T/cooperative_groups_thread_group_no_free_query %s --cuda-include-path="%cuda-path/include" --extra-arg="-std=c++14"
// RUN: FileCheck %s --match-full-lines --input-file %T/cooperative_groups_thread_group_no_free_query/cooperative_groups_thread_group_no_free_query.dp.cpp

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
// CHECK:  #define test33(a) testThreadGroup(a, item_ct1)
#define test33(a) testThreadGroup(a)
#define test22(a) test33(a)
#define test11(a) test22(a)
// CHECK:  #define test44(a) testThreadGroup(a, item_ct1)
#define test44(a) testThreadGroup(a)

namespace cg = cooperative_groups;

// CHECK:  void testThreadGroup(dpct::experimental::group_base<3> g, const sycl::nd_item<3> &item_ct1) {
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
  // CHECK:  auto threadBlockGroup = item_ct1.get_group();
  auto threadBlockGroup = cg::this_thread_block();


  testThreadGroup(threadBlockGroup);
  
  cg::thread_block_tile<16> tilePartition16 = cg::tiled_partition<16>(threadBlockGroup);
  testThreadGroup(tilePartition16);
  cg::thread_block_tile<32> tilePartition32 = cg::tiled_partition<32>(threadBlockGroup);
  testThreadGroup(tilePartition32);
  cg::thread_block_tile<16> tilePartition16_1(cg::tiled_partition<16>(threadBlockGroup));
  cg::thread_block_tile<32> tilePartition32_2(cg::tiled_partition<32>(threadBlockGroup));

  test11(tilePartition16);
  testThreadGroup(tilePartition16);
  // CHECK:  test44(dpct::experimental::group(tilePartition16, item_ct1));
  // CHECK:  test11(dpct::experimental::group(tilePartition32, item_ct1));
  // CHECK:  test11(dpct::experimental::group(threadBlockGroup, item_ct1));
  test44(tilePartition16);
  test11(tilePartition32);
  test11(threadBlockGroup);
}


int main() {
  kernelFunc<<<1,1>>>();
  return 0;
}
