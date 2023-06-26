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
  // CHECK-NEXT:  DPCT1007:{{[0-9]+}}: Migration of cooperative_groups::__v1::thread_group.thread_rank is not supported.
  // CHECK-NEXT:  */
  g.thread_rank();
  // CHECK:  /*
  // CHECK-NEXT:  DPCT1007:{{[0-9]+}}: Migration of cooperative_groups::__v1::thread_group.sync is not supported.
  // CHECK-NEXT:  */
  g.sync();
  // CHECK:  /*
  // CHECK-NEXT:  DPCT1007:{{[0-9]+}}: Migration of cooperative_groups::__v1::thread_group.size is not supported.
  // CHECK-NEXT:  */
  g.size();
}
