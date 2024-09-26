// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct -in-root %S -out-root %T/intrinsic/load %S/load.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/intrinsic/load/load.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/intrinsic/load/load.dp.cpp -o %T/intrinsic/load/load.dp.o %}

// CHECK:#include <sycl/sycl.hpp>
// CHECK:#include <dpct/dpct.hpp>
// CHECK:#include <dpct/group_utils.hpp>
#include <cub/cub.cuh>

__global__ void TestLoadStriped(int *d_data) {
  int thread_data[4];
  // CHECK: dpct::group::load_direct_striped(item_ct1, d_data, thread_data);
  cub::LoadDirectStriped<128>(threadIdx.x, d_data, thread_data);
}

__global__ void BlockedToStripedKernel(int *d_data) {
  int thread_data[4];
  // CHECK: dpct::group::load_direct_blocked(item_ct1, d_data, thread_data);
  cub::LoadDirectBlocked(threadIdx.x, d_data, thread_data);
}
