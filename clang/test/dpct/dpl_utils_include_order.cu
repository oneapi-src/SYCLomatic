// RUN: dpct --format-range=none -out-root %T/dpl_utils_include_order %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/dpl_utils_include_order/dpl_utils_include_order.dp.cpp %s

// CHECK: #include <oneapi/dpl/execution>
// CHECK-NEXT: #include <oneapi/dpl/algorithm>
// CHECK-NEXT: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <dpct/dpl_utils.hpp>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define DATA_NUM 100

bool test_dpl_utils() {
  static const int n = 10;
  int *d_in = nullptr;
  int *d_out = nullptr;
  int *d_flagged = nullptr;
  int *d_select_num = nullptr;
  int *d_tmp = nullptr;
  size_t n_d_tmp = 0;
  cub::DeviceSelect::Flagged(d_tmp, n_d_tmp, d_in,
                             d_flagged, d_out, d_select_num,
                             n);
  return true;
}
