// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -out-root %T/sync_api_ndrange_barrier %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/sync_api_ndrange_barrier/sync_api_ndrange_barrier.dp.cpp

// CHECK: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
#include "cooperative_groups.h"
namespace cg = cooperative_groups;
using namespace cooperative_groups;

// CHECK: void kernel(const sycl::stream &stream_ct1) {
// CHECK-NEXT:  /*
// CHECK-NEXT:  DPCT1119:{{[0-9]+}}: Migration of cooperative_groups::__v1::grid_group::this_grid is not supported, please try to remigrate with option: --use-experimental-features=root-group.
// CHECK-NEXT:  */
// CHECK-NEXT:  /*
// CHECK-NEXT:  DPCT1119:{{[0-9]+}}: Migration of cooperative_groups::__v1::grid_group is not supported, please try to remigrate with option: --use-experimental-features=root-group.
// CHECK-NEXT:  */
// CHECK-NEXT:  cg::grid_group grid = cg::this_grid();
// CHECK-NEXT:  stream_ct1 << "kernel run!\n";
// CHECK-NEXT:  /*
// CHECK-NEXT:  DPCT1087:{{[0-9]+}}: SYCL currently does not support cross group synchronization. You can specify "--use-experimental-features=root-group" to use the root-group to migrate grid.sync().
// CHECK-NEXT:  */
// CHECK-NEXT:  grid.sync();
// CHECK-NEXT:}
__global__ void kernel() {
  cg::grid_group grid = cg::this_grid();
  printf("kernel run!\n");
  grid.sync();
}

int main() {
// CHECK:  dpct::get_in_order_queue().submit(
// CHECK-NEXT:    [&](sycl::handler &cgh) {
// CHECK-NEXT:      sycl::stream stream_ct1(64 * 1024, 80, cgh);
// CHECK-EMPTY:
// CHECK-NEXT:      cgh.parallel_for(
// CHECK-NEXT:        sycl::nd_range<3>(sycl::range<3>(1, 1, 2) * sycl::range<3>(1, 1, 2), sycl::range<3>(1, 1, 2)), 
// CHECK-NEXT:        [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:          kernel(stream_ct1);
// CHECK-NEXT:        });
// CHECK-NEXT:    });
  kernel<<<2, 2>>>();

  cudaDeviceSynchronize();
  return 0;
}