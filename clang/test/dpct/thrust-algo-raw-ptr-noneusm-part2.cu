// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --usm-level=none -out-root %T/thrust-algo-raw-ptr-noneusm-part2 %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17 -fsized-deallocation
// RUN: FileCheck --input-file %T/thrust-algo-raw-ptr-noneusm-part2/thrust-algo-raw-ptr-noneusm-part2.dp.cpp --match-full-lines %s

#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/set_operations.h>

struct Compare {
  __host__ __device__ bool operator()(const int &a, const int &b) {
    return a < b;
  }
};

void set_symmetric_difference() {
  int A1[7] = {0, 1, 2, 2, 4, 6, 7};
  int A2[5] = {1, 1, 2, 5, 8};
  int result[8];

  // CHECK:  if (dpct::is_device_ptr(A1)) {
  // CHECK-NEXT:    oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(A1), dpct::device_pointer<int>(A1 + 4), dpct::device_pointer<int>(A2), dpct::device_pointer<int>(A2 + 2), dpct::device_pointer<int>(result));
  // CHECK-NEXT:  } else {
  // CHECK-NEXT:    oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::seq, A1, A1 + 4, A2, A2 + 2, result);
  // CHECK-NEXT:  };
  // CHECK-NEXT:  if (dpct::is_device_ptr(A1)) {
  // CHECK-NEXT:    oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(A1), dpct::device_pointer<int>(A1 + 4), dpct::device_pointer<int>(A2), dpct::device_pointer<int>(A2 + 2), dpct::device_pointer<int>(result));
  // CHECK-NEXT:  } else {
  // CHECK-NEXT:    oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::seq, A1, A1 + 4, A2, A2 + 2, result);
  // CHECK-NEXT:  };
  // CHECK-NEXT:  if (dpct::is_device_ptr(A1)) {
  // CHECK-NEXT:    oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(A1), dpct::device_pointer<int>(A1 + 5), dpct::device_pointer<int>(A2), dpct::device_pointer<int>(A2 + 5), dpct::device_pointer<int>(result), Compare());
  // CHECK-NEXT:  } else {
  // CHECK-NEXT:    oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::seq, A1, A1 + 5, A2, A2 + 5, result, Compare());
  // CHECK-NEXT:  };
  // CHECK-NEXT:  if (dpct::is_device_ptr(A1)) {
  // CHECK-NEXT:    oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(A1), dpct::device_pointer<int>(A1 + 5), dpct::device_pointer<int>(A2), dpct::device_pointer<int>(A2 + 5), dpct::device_pointer<int>(result), Compare());
  // CHECK-NEXT:  } else {
  // CHECK-NEXT:    oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::seq, A1, A1 + 5, A2, A2 + 5, result, Compare());
  // CHECK-NEXT:  };
  thrust::set_symmetric_difference(thrust::host, A1, A1 + 4, A2, A2 + 2, result);
  thrust::set_symmetric_difference(A1, A1 + 4, A2, A2 + 2, result);
  thrust::set_symmetric_difference(thrust::host, A1, A1 + 5, A2, A2 + 5, result, Compare());
  thrust::set_symmetric_difference(A1, A1 + 5, A2, A2 + 5, result, Compare());
}
