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
#include <thrust/sort.h>
#include <thrust/set_operations.h>


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

void is_sorted_until() {
  int A[8] = {0, 1, 2, 3, 0, 1, 2, 3};
  thrust::host_vector<int> h_A(A, A + 8);
  thrust::device_vector<int> d_A(A, A + 8);
  thrust::greater<int> comp;
  int *B;
  thrust::host_vector<int>::iterator h_end;
  thrust::device_vector<int>::iterator d_end;

  // CHECK:B = oneapi::dpl::is_sorted_until(oneapi::dpl::execution::seq, A, A + 8);
  // CHECK-NEXT:B = oneapi::dpl::is_sorted_until(oneapi::dpl::execution::seq, A, A + 8);
  // CHECK-NEXT:B = oneapi::dpl::is_sorted_until(oneapi::dpl::execution::seq, A, A + 8, comp);
  // CHECK-NEXT:B = oneapi::dpl::is_sorted_until(oneapi::dpl::execution::seq, A, A + 8, comp);
  // CHECK-NEXT:h_end = oneapi::dpl::is_sorted_until(oneapi::dpl::execution::seq, h_A.begin(), h_A.end());
  // CHECK-NEXT:h_end = oneapi::dpl::is_sorted_until(oneapi::dpl::execution::seq, h_A.begin(), h_A.end());
  // CHECK-NEXT:h_end = oneapi::dpl::is_sorted_until(oneapi::dpl::execution::seq, h_A.begin(), h_A.end(), comp);
  // CHECK-NEXT:h_end = oneapi::dpl::is_sorted_until(oneapi::dpl::execution::seq, h_A.begin(), h_A.end(), comp);
  // CHECK-NEXT:d_end = oneapi::dpl::is_sorted_until(oneapi::dpl::execution::make_device_policy(q_ct1), d_A.begin(), d_A.end());
  // CHECK-NEXT:d_end = oneapi::dpl::is_sorted_until(oneapi::dpl::execution::make_device_policy(q_ct1), d_A.begin(), d_A.end());
  // CHECK-NEXT:d_end = oneapi::dpl::is_sorted_until(oneapi::dpl::execution::make_device_policy(q_ct1), d_A.begin(), d_A.end(), comp);
  // CHECK-NEXT:d_end = oneapi::dpl::is_sorted_until(oneapi::dpl::execution::make_device_policy(q_ct1), d_A.begin(), d_A.end(), comp);
  B = thrust::is_sorted_until(thrust::host, A, A + 8);
  B = thrust::is_sorted_until(A, A + 8);
  B = thrust::is_sorted_until(thrust::host, A, A + 8, comp);
  B = thrust::is_sorted_until(A, A + 8, comp);
  h_end = thrust::is_sorted_until(thrust::host, h_A.begin(), h_A.end());
  h_end = thrust::is_sorted_until(h_A.begin(), h_A.end());
  h_end = thrust::is_sorted_until(thrust::host, h_A.begin(), h_A.end(), comp);
  h_end = thrust::is_sorted_until(h_A.begin(), h_A.end(), comp);
  d_end = thrust::is_sorted_until(thrust::device, d_A.begin(), d_A.end());
  d_end = thrust::is_sorted_until(d_A.begin(), d_A.end());
  d_end = thrust::is_sorted_until(thrust::device, d_A.begin(), d_A.end(), comp);
  d_end = thrust::is_sorted_until(d_A.begin(), d_A.end(), comp);
}

void set_intersection() {
  int A1[6] = {1, 3, 5, 7, 9, 11};
  int A2[7] = {1, 1, 2, 3, 5, 8, 13};
  int result[3];
  int *result_end;
  thrust::device_vector<int> d_A1(A1, A1 + 6);
  thrust::device_vector<int> d_A2(A2, A2 + 7);
  thrust::device_vector<int> d_result(3);
  thrust::device_vector<int>::iterator d_end;
  thrust::host_vector<int> h_A1(A1, A1 + 6);
  thrust::host_vector<int> h_A2(A2, A2 + 7);
  thrust::host_vector<int> h_result(3);
  thrust::host_vector<int>::iterator h_end;

  result_end = thrust::set_intersection(thrust::host, A1, A1 + 6, A2, A2 + 7, result);
  result_end = thrust::set_intersection(A1, A1 + 6, A2, A2 + 7, result);
  result_end = thrust::set_intersection(thrust::host, A1, A1 + 6, A2, A2 + 7, result, thrust::greater<int>());
  result_end = thrust::set_intersection(A1, A1 + 6, A2, A2 + 7, result, thrust::greater<int>());

  d_end = thrust::set_intersection(thrust::device, d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin());
  d_end = thrust::set_intersection(d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin());
  d_end = thrust::set_intersection(thrust::device, d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin(), thrust::greater<int>());
  d_end = thrust::set_intersection(d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin(), thrust::greater<int>());
  h_end = thrust::set_intersection(thrust::host, h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin());
  h_end = thrust::set_intersection(h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin());
  h_end = thrust::set_intersection(thrust::host, h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin(), thrust::greater<int>());
  h_end = thrust::set_intersection(h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin(), thrust::greater<int>());
}
