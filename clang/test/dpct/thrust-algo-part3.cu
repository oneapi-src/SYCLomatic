// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -out-root %T/thrust-algo-part3 %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/thrust-algo-part3/thrust-algo-part3.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/thrust-algo-part3/thrust-algo-part3.dp.cpp -o %T/thrust-algo-part3/thrust-algo-part3.dp.o %}

#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>
#include <thrust/partition.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>

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

  // CHECK:  result_end = oneapi::dpl::set_intersection(oneapi::dpl::execution::seq, A1, A1 + 6, A2, A2 + 7, result);
  // CHECK-NEXT:  result_end = oneapi::dpl::set_intersection(oneapi::dpl::execution::seq, A1, A1 + 6, A2, A2 + 7, result);
  // CHECK-NEXT:  result_end = oneapi::dpl::set_intersection(oneapi::dpl::execution::seq, A1, A1 + 6, A2, A2 + 7, result, std::greater<int>());
  // CHECK-NEXT:  result_end = oneapi::dpl::set_intersection(oneapi::dpl::execution::seq, A1, A1 + 6, A2, A2 + 7, result, std::greater<int>());
  // CHECK-NEXT:   d_end = oneapi::dpl::set_intersection(oneapi::dpl::execution::make_device_policy(q_ct1), d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin());
  // CHECK-NEXT:   d_end = oneapi::dpl::set_intersection(oneapi::dpl::execution::make_device_policy(q_ct1), d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin());
  // CHECK-NEXT:   d_end = oneapi::dpl::set_intersection(oneapi::dpl::execution::make_device_policy(q_ct1), d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin(), std::greater<int>());
  // CHECK-NEXT:   d_end = oneapi::dpl::set_intersection(oneapi::dpl::execution::make_device_policy(q_ct1), d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin(), std::greater<int>());
  // CHECK-NEXT:   h_end = oneapi::dpl::set_intersection(oneapi::dpl::execution::seq, h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin());
  // CHECK-NEXT:   h_end = oneapi::dpl::set_intersection(oneapi::dpl::execution::seq, h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin());
  // CHECK-NEXT:  h_end = oneapi::dpl::set_intersection(oneapi::dpl::execution::seq, h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin(), std::greater<int>());
  // CHECK-NEXT:  h_end = oneapi::dpl::set_intersection(oneapi::dpl::execution::seq, h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin(), std::greater<int>());
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

void set_union() {

  int A1[7] = {0, 2, 4, 6, 8, 10, 12};
  int A2[5] = {1, 3, 5, 7, 9};
  int result[12];
  int *result_end;
  thrust::device_vector<int> d_A1(A1, A1 + 7);
  thrust::device_vector<int> d_A2(A2, A2 + 5);
  thrust::device_vector<int> d_result(11);
  thrust::device_vector<int>::iterator d_result_iter_end;
  thrust::host_vector<int> h_A1(A1, A1 + 7);
  thrust::host_vector<int> h_A2(A2, A2 + 5);
  thrust::host_vector<int> h_result(12);
  thrust::host_vector<int>::iterator h_result_iter_end;

  // CHECK:  result_end = oneapi::dpl::set_union(oneapi::dpl::execution::seq, A1, A1 + 7, A2, A2 + 5, result);
  // CHECK-NEXT:  result_end = oneapi::dpl::set_union(oneapi::dpl::execution::seq, A1, A1 + 7, A2, A2 + 5, result);
  // CHECK-NEXT:  result_end = oneapi::dpl::set_union(oneapi::dpl::execution::seq, A1, A1 + 7, A2, A2 + 5, result, std::greater<int>());
  // CHECK-NEXT:  result_end = oneapi::dpl::set_union(oneapi::dpl::execution::seq, A1, A1 + 7, A2, A2 + 5, result, std::greater<int>());
  // CHECK-NEXT:  d_result_iter_end = oneapi::dpl::set_union(oneapi::dpl::execution::make_device_policy(q_ct1), d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin());
  // CHECK-NEXT:  d_result_iter_end = oneapi::dpl::set_union(oneapi::dpl::execution::make_device_policy(q_ct1), d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin());
  // CHECK-NEXT:  d_result_iter_end = oneapi::dpl::set_union(oneapi::dpl::execution::make_device_policy(q_ct1), d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin(), std::greater<int>());
  // CHECK-NEXT:  d_result_iter_end = oneapi::dpl::set_union(oneapi::dpl::execution::make_device_policy(q_ct1), d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin(), std::greater<int>());
  // CHECK-NEXT:  h_result_iter_end = oneapi::dpl::set_union(oneapi::dpl::execution::seq, h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin());
  // CHECK-NEXT:  h_result_iter_end = oneapi::dpl::set_union(oneapi::dpl::execution::seq, h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin());
  // CHECK-NEXT:  h_result_iter_end = oneapi::dpl::set_union(oneapi::dpl::execution::seq, h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin(), std::greater<int>());
  // CHECK-NEXT:  h_result_iter_end = oneapi::dpl::set_union(oneapi::dpl::execution::seq, h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin(), std::greater<int>());
  result_end = thrust::set_union(thrust::host, A1, A1 + 7, A2, A2 + 5, result);
  result_end = thrust::set_union(A1, A1 + 7, A2, A2 + 5, result);
  result_end = thrust::set_union(thrust::host, A1, A1 + 7, A2, A2 + 5, result, thrust::greater<int>());
  result_end = thrust::set_union(A1, A1 + 7, A2, A2 + 5, result, thrust::greater<int>());
  d_result_iter_end = thrust::set_union(thrust::device, d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin());
  d_result_iter_end = thrust::set_union(d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin());
  d_result_iter_end = thrust::set_union(thrust::device, d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin(), thrust::greater<int>());
  d_result_iter_end = thrust::set_union(d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin(), thrust::greater<int>());
  h_result_iter_end = thrust::set_union(thrust::host, h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin());
  h_result_iter_end = thrust::set_union(h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin());
  h_result_iter_end = thrust::set_union(thrust::host, h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin(), thrust::greater<int>());
  h_result_iter_end = thrust::set_union(h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin(), thrust::greater<int>());
}

struct Compare {
  __host__ __device__ bool operator()(const int &a, const int &b) {
    // Custom comparison function
    // Returns true if a < b, false otherwise
    return a < b;
  }
};

void set_union_by_key() {

  int A_keys[7] = {0, 2, 4};
  int A_vals[7] = {0, 0, 0};
  int B_keys[5] = {0, 3, 3, 4};
  int B_vals[5] = {1, 1, 1, 1};
  int keys_result[5];
  int vals_result[5];
  thrust::pair<int *, int *> end;
  thrust::device_vector<int> d_keys_result(10);
  thrust::device_vector<int> d_vals_result(10);

  thrust::device_vector<int> d_A_keys(A_keys, A_keys + 7);
  thrust::device_vector<int> d_A_vals(A_vals, A_vals + 7);
  thrust::device_vector<int> d_B_keys(B_keys, B_keys + 5);
  thrust::device_vector<int> d_B_vals(B_vals, B_vals + 5);
  typedef thrust::device_vector<int>::iterator d_Iterator;
  thrust::pair<d_Iterator, d_Iterator> d_result;
  thrust::host_vector<int> h_keys_result(10);
  thrust::host_vector<int> h_vals_result(10);
  thrust::host_vector<int> h_A_keys(A_keys, A_keys + 7);
  thrust::host_vector<int> h_A_vals(A_vals, A_vals + 7);
  thrust::host_vector<int> h_B_keys(B_keys, B_keys + 5);
  thrust::host_vector<int> h_B_vals(B_vals, B_vals + 5);
  typedef thrust::host_vector<int>::iterator h_Iterator;
  thrust::pair<h_Iterator, h_Iterator> h_result;

  // CHECK:  end = dpct::set_union(oneapi::dpl::execution::seq, A_keys, A_keys + 3, B_keys, B_keys + 4, A_vals, B_vals, keys_result, vals_result);
  // CHECK-NEXT:  end = dpct::set_union(oneapi::dpl::execution::seq, A_keys, A_keys + 3, B_keys, B_keys + 4, A_vals, B_vals, keys_result, vals_result);
  // CHECK-NEXT:  end = dpct::set_union(oneapi::dpl::execution::seq, A_keys, A_keys + 7, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result, Compare());
  // CHECK-NEXT:  end = dpct::set_union(oneapi::dpl::execution::seq, A_keys, A_keys + 7, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result, Compare());
  // CHECK-NEXT:  d_result = dpct::set_union(oneapi::dpl::execution::make_device_policy(q_ct1), d_A_keys.begin(), d_A_keys.end(), d_B_keys.begin(), d_B_keys.end(), d_A_vals.begin(), d_B_vals.begin(), d_keys_result.begin(), d_vals_result.begin());
  // CHECK-NEXT:  d_result = dpct::set_union(oneapi::dpl::execution::make_device_policy(q_ct1), d_A_keys.begin(), d_A_keys.end(), d_B_keys.begin(), d_B_keys.end(), d_A_vals.begin(), d_B_vals.begin(), d_keys_result.begin(), d_vals_result.begin());
  // CHECK-NEXT:  d_result = dpct::set_union(oneapi::dpl::execution::make_device_policy(q_ct1), d_A_keys.begin(), d_A_keys.end(), d_B_keys.begin(), d_B_keys.end(), d_A_vals.begin(), d_B_vals.begin(), d_keys_result.begin(), d_vals_result.begin(), Compare());
  // CHECK-NEXT:  d_result = dpct::set_union(oneapi::dpl::execution::make_device_policy(q_ct1), d_A_keys.begin(), d_A_keys.end(), d_B_keys.begin(), d_B_keys.end(), d_A_vals.begin(), d_B_vals.begin(), d_keys_result.begin(), d_vals_result.begin(), Compare());
  // CHECK-NEXT:  h_result = dpct::set_union(oneapi::dpl::execution::seq, h_A_keys.begin(), h_A_keys.end(), h_B_keys.begin(), h_B_keys.end(), h_A_vals.begin(), h_B_vals.begin(), h_keys_result.begin(), h_vals_result.begin());
  // CHECK-NEXT:  h_result = dpct::set_union(oneapi::dpl::execution::seq, h_A_keys.begin(), h_A_keys.end(), h_B_keys.begin(), h_B_keys.end(), h_A_vals.begin(), h_B_vals.begin(), h_keys_result.begin(), h_vals_result.begin());
  // CHECK-NEXT:  h_result = dpct::set_union(oneapi::dpl::execution::seq, h_A_keys.begin(), h_A_keys.end(), h_B_keys.begin(), h_B_keys.end(), h_A_vals.begin(), h_B_vals.begin(), h_keys_result.begin(), h_vals_result.begin(), Compare());
  // CHECK-NEXT:  h_result = dpct::set_union(oneapi::dpl::execution::seq, h_A_keys.begin(), h_A_keys.end(), h_B_keys.begin(), h_B_keys.end(), h_A_vals.begin(), h_B_vals.begin(), h_keys_result.begin(), h_vals_result.begin(), Compare());
  end = thrust::set_union_by_key(thrust::host, A_keys, A_keys + 3, B_keys, B_keys + 4, A_vals, B_vals, keys_result, vals_result);
  end = thrust::set_union_by_key(A_keys, A_keys + 3, B_keys, B_keys + 4, A_vals, B_vals, keys_result, vals_result);
  end = thrust::set_union_by_key(thrust::host, A_keys, A_keys + 7, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result, Compare());
  end = thrust::set_union_by_key(A_keys, A_keys + 7, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result, Compare());
  d_result = thrust::set_union_by_key(thrust::device, d_A_keys.begin(), d_A_keys.end(), d_B_keys.begin(), d_B_keys.end(), d_A_vals.begin(), d_B_vals.begin(), d_keys_result.begin(), d_vals_result.begin());
  d_result = thrust::set_union_by_key(d_A_keys.begin(), d_A_keys.end(), d_B_keys.begin(), d_B_keys.end(), d_A_vals.begin(), d_B_vals.begin(), d_keys_result.begin(), d_vals_result.begin());
  d_result = thrust::set_union_by_key(thrust::device, d_A_keys.begin(), d_A_keys.end(), d_B_keys.begin(), d_B_keys.end(), d_A_vals.begin(), d_B_vals.begin(), d_keys_result.begin(), d_vals_result.begin(), Compare());
  d_result = thrust::set_union_by_key(d_A_keys.begin(), d_A_keys.end(), d_B_keys.begin(), d_B_keys.end(), d_A_vals.begin(), d_B_vals.begin(), d_keys_result.begin(), d_vals_result.begin(), Compare());
  h_result = thrust::set_union_by_key(thrust::host, h_A_keys.begin(), h_A_keys.end(), h_B_keys.begin(), h_B_keys.end(), h_A_vals.begin(), h_B_vals.begin(), h_keys_result.begin(), h_vals_result.begin());
  h_result = thrust::set_union_by_key(h_A_keys.begin(), h_A_keys.end(), h_B_keys.begin(), h_B_keys.end(), h_A_vals.begin(), h_B_vals.begin(), h_keys_result.begin(), h_vals_result.begin());
  h_result = thrust::set_union_by_key(thrust::host, h_A_keys.begin(), h_A_keys.end(), h_B_keys.begin(), h_B_keys.end(), h_A_vals.begin(), h_B_vals.begin(), h_keys_result.begin(), h_vals_result.begin(), Compare());
  h_result = thrust::set_union_by_key(h_A_keys.begin(), h_A_keys.end(), h_B_keys.begin(), h_B_keys.end(), h_A_vals.begin(), h_B_vals.begin(), h_keys_result.begin(), h_vals_result.begin(), Compare());
}


void reverse_copy() {
  const int N = 6;
  int data[N] = {0, 1, 2, 3, 4, 5};
  thrust::device_vector<int> device_data(data, data + N);
  thrust::host_vector<int> host_data(data, data + N);
  thrust::device_vector<int> host_result(N);
  thrust::device_vector<int> device_result(N);
  int result[N];

  // CHECK:  oneapi::dpl::reverse_copy(oneapi::dpl::execution::make_device_policy(q_ct1), device_data.begin(), device_data.end(), device_result.begin());
  // CHECK-NEXT:  oneapi::dpl::reverse_copy(oneapi::dpl::execution::seq, host_data.begin(), host_data.end(), host_result.begin());
  // CHECK-NEXT:  oneapi::dpl::reverse_copy(oneapi::dpl::execution::seq, data, data + N, result);
  // CHECK-NEXT:  oneapi::dpl::reverse_copy(oneapi::dpl::execution::make_device_policy(q_ct1), device_data.begin(), device_data.end(), device_result.begin());
  // CHECK-NEXT:  oneapi::dpl::reverse_copy(oneapi::dpl::execution::seq, host_data.begin(), host_data.end(), host_result.begin());
  // CHECK-NEXT:  oneapi::dpl::reverse_copy(oneapi::dpl::execution::seq, data, data + N, result);
  thrust::reverse_copy(thrust::device, device_data.begin(), device_data.end(), device_result.begin());
  thrust::reverse_copy(thrust::host, host_data.begin(), host_data.end(), host_result.begin());
  thrust::reverse_copy(thrust::host, data, data + N, result);
  thrust::reverse_copy(device_data.begin(), device_data.end(), device_result.begin());
  thrust::reverse_copy(host_data.begin(), host_data.end(), host_result.begin());
  thrust::reverse_copy(data, data + N, result);
}