// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -out-root %T/thrust-algo-part3 %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/thrust-algo-part3/thrust-algo-part3.dp.cpp --match-full-lines %s

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>
#include <thrust/partition.h>

void all_of() {

  bool A[3] = {true, true, false};
  bool result;
  thrust::host_vector<bool> h_A(A, A + 3);
  thrust::device_vector<bool> d_A(A, A + 3);

  // CHECK:  result = oneapi::dpl::all_of(oneapi::dpl::execution::seq, A, A + 2, oneapi::dpl::identity());
  // CHECK-NEXT:  result = oneapi::dpl::all_of(oneapi::dpl::execution::seq, A, A + 2, oneapi::dpl::identity());
  // CHECK-NEXT:  result = oneapi::dpl::all_of(oneapi::dpl::execution::seq, h_A.begin(), h_A.begin() + 2, oneapi::dpl::identity());
  // CHECK-NEXT:  result = oneapi::dpl::all_of(oneapi::dpl::execution::seq, h_A.begin(), h_A.begin() + 2, oneapi::dpl::identity());
  // CHECK-NEXT:  result = oneapi::dpl::all_of(oneapi::dpl::execution::make_device_policy(q_ct1), d_A.begin(), d_A.begin() + 2, oneapi::dpl::identity());
  // CHECK-NEXT:  result = oneapi::dpl::all_of(oneapi::dpl::execution::make_device_policy(q_ct1), d_A.begin(), d_A.begin() + 2, oneapi::dpl::identity());
  result = thrust::all_of(thrust::host, A, A + 2, thrust::identity<bool>());
  result = thrust::all_of(A, A + 2, thrust::identity<bool>());
  result = thrust::all_of(thrust::host, h_A.begin(), h_A.begin() + 2, thrust::identity<bool>());
  result = thrust::all_of(h_A.begin(), h_A.begin() + 2, thrust::identity<bool>());
  result = thrust::all_of(thrust::device, d_A.begin(), d_A.begin() + 2, thrust::identity<bool>());
  result = thrust::all_of(d_A.begin(), d_A.begin() + 2, thrust::identity<bool>());
}

void none_of() {
  bool A[3] = {true, true, false};
  thrust::host_vector<bool> h_A(A, A + 3);
  thrust::device_vector<bool> d_A(A, A + 3);
  bool result;

  // CHECK:  result = oneapi::dpl::none_of(oneapi::dpl::execution::seq, A, A + 2, oneapi::dpl::identity());
  // CHECK-NEXT:  result = oneapi::dpl::none_of(oneapi::dpl::execution::seq, A, A + 2, oneapi::dpl::identity());
  // CHECK-NEXT:  result = oneapi::dpl::none_of(oneapi::dpl::execution::seq, h_A.begin(), h_A.begin() + 2, oneapi::dpl::identity());
  // CHECK-NEXT:  result = oneapi::dpl::none_of(oneapi::dpl::execution::seq, h_A.begin(), h_A.begin() + 2, oneapi::dpl::identity());
  // CHECK-NEXT:  result = oneapi::dpl::none_of(oneapi::dpl::execution::make_device_policy(q_ct1), d_A.begin(), d_A.begin() + 2, oneapi::dpl::identity());
  // CHECK-NEXT:  result = oneapi::dpl::none_of(oneapi::dpl::execution::make_device_policy(q_ct1), d_A.begin(), d_A.begin() + 2, oneapi::dpl::identity());
  result = thrust::none_of(thrust::host, A, A + 2, thrust::identity<bool>());
  result = thrust::none_of(A, A + 2, thrust::identity<bool>());
  result = thrust::none_of(thrust::host, h_A.begin(), h_A.begin() + 2, thrust::identity<bool>());
  result = thrust::none_of(h_A.begin(), h_A.begin() + 2, thrust::identity<bool>());
  result = thrust::none_of(thrust::device, d_A.begin(), d_A.begin() + 2, thrust::identity<bool>());
  result = thrust::none_of(d_A.begin(), d_A.begin() + 2, thrust::identity<bool>());
}

struct is_even {
  __host__ __device__ bool operator()(const int &x) const { return (x % 2) == 0; }
};

void is_partitioned() {

  int A[] = {2, 4, 6, 8, 10, 1, 3, 5, 7, 9};
  bool result;
  thrust::host_vector<int> h_A(A, A + 10);
  thrust::device_vector<int> d_A(A, A + 10);

  // CHECK:  result = oneapi::dpl::is_partitioned(oneapi::dpl::execution::seq, A, A + 10, is_even());
  // CHECK-NEXT:  result = oneapi::dpl::is_partitioned(oneapi::dpl::execution::seq, A, A + 10, is_even());
  // CHECK-NEXT:  result = oneapi::dpl::is_partitioned(oneapi::dpl::execution::seq, h_A.begin(), h_A.end(), is_even());
  // CHECK-NEXT:  result = oneapi::dpl::is_partitioned(oneapi::dpl::execution::seq, h_A.begin(), h_A.end(), is_even());
  // CHECK-NEXT:  result = oneapi::dpl::is_partitioned(oneapi::dpl::execution::make_device_policy(q_ct1), d_A.begin(), d_A.end(), is_even());
  // CHECK-NEXT:  result = oneapi::dpl::is_partitioned(oneapi::dpl::execution::make_device_policy(q_ct1), d_A.begin(), d_A.end(), is_even());
  result = thrust::is_partitioned(thrust::host, A, A + 10, is_even());
  result = thrust::is_partitioned(A, A + 10, is_even());
  result = thrust::is_partitioned(thrust::host, h_A.begin(), h_A.end(), is_even());
  result = thrust::is_partitioned(h_A.begin(), h_A.end(), is_even());
  result = thrust::is_partitioned(thrust::device, d_A.begin(), d_A.end(), is_even());
  result = thrust::is_partitioned(d_A.begin(), d_A.end(), is_even());
}
