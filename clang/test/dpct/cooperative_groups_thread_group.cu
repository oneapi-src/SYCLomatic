// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0
// RUN: dpct --format-range=none  -use-experimental-features=free-function-queries,logical-group -out-root %T/cooperative_groups_thread_group %s --cuda-include-path="%cuda-path/include" --extra-arg="-std=c++14"
// RUN: FileCheck %s --match-full-lines --input-file %T/cooperative_groups_thread_group/cooperative_groups_thread_group.dp.cpp

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;
// CHECK:  /*
// CHECK-NEXT:  DPCT1082:{{[0-9]+}}: Migration of cg::thread_group type is not supported.
// CHECK-NEXT:  */
__device__ void testThreadGroup(cg::thread_group g) {
  // CHECK:  /*
  // CHECK-NEXT:  DPCT1007:{{[0-9]+}}: Migration of cooperative_groups::thread_group::thread_rank is not supported.
  // CHECK-NEXT:  */
  g.thread_rank();
  // CHECK:  /*
  // CHECK-NEXT:  DPCT1007:{{[0-9]+}}: Migration of cooperative_groups::thread_group::sync is not supported.
  // CHECK-NEXT:  */
  g.sync();
  // CHECK:  /*
  // CHECK-NEXT:  DPCT1007:{{[0-9]+}}: Migration of cooperative_groups::thread_group::size is not supported.
  // CHECK-NEXT:  */
  g.size();
}

__global__ void testThreadBlock() {
  auto block = cg::this_thread_block();
  // CHECK: sycl::ext::oneapi::experimental::this_nd_item<3>().get_local_id();
  block.thread_index();
}
