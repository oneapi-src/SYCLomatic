// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -out-root %T/thrust-algo-part2 %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/thrust-algo-part2/thrust-algo-part2.dp.cpp --match-full-lines %s

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
  thrust::device_vector<int> d_A1(A1, A1 + 7);
  thrust::device_vector<int> d_A2(A2, A2 + 5);
  thrust::device_vector<int> d_result(8);
  thrust::host_vector<int> h_A1(A1, A1 + 7);
  thrust::host_vector<int> h_A2(A2, A2 + 5);
  thrust::host_vector<int> h_result(8);

  // CHECK:  oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::seq, A1, A1 + 4, A2, A2 + 2, result);
  // CHECK-NEXT:  oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::seq, A1, A1 + 4, A2, A2 + 2, result);
  // CHECK-NEXT:  oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::seq, A1, A1 + 5, A2, A2 + 5, result, Compare());
  // CHECK-NEXT:  oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::seq, A1, A1 + 5, A2, A2 + 5, result, Compare());
  // CHECK-NEXT:  oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::make_device_policy(q_ct1), d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin());
  // CHECK-NEXT:  oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::make_device_policy(q_ct1), d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin());
  // CHECK-NEXT:  oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::make_device_policy(q_ct1), d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin(), oneapi::dpl::less<int>());
  // CHECK-NEXT:  oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::make_device_policy(q_ct1), d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin(), oneapi::dpl::less<int>());
  // CHECK-NEXT:  oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::seq, h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin());
  // CHECK-NEXT:  oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::seq, h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin());
  // CHECK-NEXT:  oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::seq, h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin(), oneapi::dpl::less<int>());
  // CHECK-NEXT:  oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::seq, h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin(), oneapi::dpl::less<int>());
  thrust::set_symmetric_difference(thrust::host, A1, A1 + 4, A2, A2 + 2, result);
  thrust::set_symmetric_difference(A1, A1 + 4, A2, A2 + 2, result);
  thrust::set_symmetric_difference(thrust::host, A1, A1 + 5, A2, A2 + 5, result, Compare());
  thrust::set_symmetric_difference(A1, A1 + 5, A2, A2 + 5, result, Compare());
  thrust::set_symmetric_difference(thrust::device, d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin());
  thrust::set_symmetric_difference(d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin());
  thrust::set_symmetric_difference(thrust::device, d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin(), thrust::less<int>());
  thrust::set_symmetric_difference(d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin(), thrust::less<int>());
  thrust::set_symmetric_difference(thrust::host, h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin());
  thrust::set_symmetric_difference(h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin());
  thrust::set_symmetric_difference(thrust::host, h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin(), thrust::less<int>());
  thrust::set_symmetric_difference(h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin(), thrust::less<int>());
}
