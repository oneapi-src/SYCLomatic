// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -out-root %T/thrust-algo-part2 %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/thrust-algo-part2/thrust-algo-part2.dp.cpp --match-full-lines %s

#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/set_operations.h>

// CHECK: #include <oneapi/dpl/memory>
#include <thrust/uninitialized_fill.h>

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

  // CHECK:  oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::par_unseq, A1, A1 + 4, A2, A2 + 2, result);
  // CHECK-NEXT:  oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::par_unseq, A1, A1 + 4, A2, A2 + 2, result);
  // CHECK-NEXT:  oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::par_unseq, A1, A1 + 5, A2, A2 + 5, result, Compare());
  // CHECK-NEXT:  oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::par_unseq, A1, A1 + 5, A2, A2 + 5, result, Compare());
  // CHECK-NEXT:  oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::make_device_policy(q_ct1), d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin());
  // CHECK-NEXT:  oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::make_device_policy(q_ct1), d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin());
  // CHECK-NEXT:  oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::make_device_policy(q_ct1), d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin(), oneapi::dpl::less<int>());
  // CHECK-NEXT:  oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::make_device_policy(q_ct1), d_A1.begin(), d_A1.end(), d_A2.begin(), d_A2.end(), d_result.begin(), oneapi::dpl::less<int>());
  // CHECK-NEXT:  oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::par_unseq, h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin());
  // CHECK-NEXT:  oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::par_unseq, h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin());
  // CHECK-NEXT:  oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::par_unseq, h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin(), oneapi::dpl::less<int>());
  // CHECK-NEXT:  oneapi::dpl::set_symmetric_difference(oneapi::dpl::execution::par_unseq, h_A1.begin(), h_A1.end(), h_A2.begin(), h_A2.end(), h_result.begin(), oneapi::dpl::less<int>());
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

void set_symmetric_difference_by_key() {
  int A_keys[7] = {0, 1, 2, 2, 4, 6, 7};
  int A_vals[7] = {0, 0, 0, 0, 0, 0, 0};
  int B_keys[5] = {1, 1, 2, 5, 8};
  int B_vals[5] = {1, 1, 1, 1, 1};
  int keys_result[8];
  int vals_result[8];
  thrust::device_vector<int> d_keys_result(8);
  thrust::device_vector<int> d_vals_result(8);
  thrust::device_vector<int> d_A_keys(A_keys, A_keys + 7);
  thrust::device_vector<int> d_A_vals(A_vals, A_vals + 7);
  thrust::device_vector<int> d_B_keys(B_keys, B_keys + 5);
  thrust::device_vector<int> d_B_vals(B_vals, B_vals + 5);
  thrust::host_vector<int> h_keys_result(8);
  thrust::host_vector<int> h_vals_result(8);
  thrust::host_vector<int> h_A_keys(A_keys, A_keys + 7);
  thrust::host_vector<int> h_A_vals(A_vals, A_vals + 7);
  thrust::host_vector<int> h_B_keys(B_keys, B_keys + 5);
  thrust::host_vector<int> h_B_vals(B_vals, B_vals + 5);

  // CHECK:   dpct::set_symmetric_difference(oneapi::dpl::execution::par_unseq, A_keys, A_keys + 7, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result);
  // CHECK-NEXT:   dpct::set_symmetric_difference(oneapi::dpl::execution::par_unseq, A_keys, A_keys + 7, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result);
  // CHECK-NEXT:   dpct::set_symmetric_difference(oneapi::dpl::execution::par_unseq, A_keys, A_keys + 7, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result, Compare());
  // CHECK-NEXT:   dpct::set_symmetric_difference(oneapi::dpl::execution::par_unseq, A_keys, A_keys + 7, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result, Compare());
  // CHECK-NEXT:   dpct::set_symmetric_difference(oneapi::dpl::execution::make_device_policy(q_ct1), d_A_keys.begin(), d_A_keys.end(), d_B_keys.begin(), d_B_keys.end(), d_A_vals.begin(), d_B_vals.begin(), d_keys_result.begin(), d_vals_result.begin());
  // CHECK-NEXT:   dpct::set_symmetric_difference(oneapi::dpl::execution::make_device_policy(q_ct1), d_A_keys.begin(), d_A_keys.end(), d_B_keys.begin(), d_B_keys.end(), d_A_vals.begin(), d_B_vals.begin(), d_keys_result.begin(), d_vals_result.begin());
  // CHECK-NEXT:   dpct::set_symmetric_difference(oneapi::dpl::execution::make_device_policy(q_ct1), d_A_keys.begin(), d_A_keys.end(), d_B_keys.begin(), d_B_keys.end(), d_A_vals.begin(), d_B_vals.begin(), d_keys_result.begin(), d_vals_result.begin(), Compare());
  // CHECK-NEXT:   dpct::set_symmetric_difference(oneapi::dpl::execution::make_device_policy(q_ct1), d_A_keys.begin(), d_A_keys.end(), d_B_keys.begin(), d_B_keys.end(), d_A_vals.begin(), d_B_vals.begin(), d_keys_result.begin(), d_vals_result.begin(), Compare());
  // CHECK-NEXT:   dpct::set_symmetric_difference(oneapi::dpl::execution::par_unseq, h_A_keys.begin(), h_A_keys.end(), h_B_keys.begin(), h_B_keys.end(), h_A_vals.begin(), h_B_vals.begin(), h_keys_result.begin(), h_vals_result.begin());
  // CHECK-NEXT:   dpct::set_symmetric_difference(oneapi::dpl::execution::par_unseq, h_A_keys.begin(), h_A_keys.end(), h_B_keys.begin(), h_B_keys.end(), h_A_vals.begin(), h_B_vals.begin(), h_keys_result.begin(), h_vals_result.begin());
  // CHECK-NEXT:   dpct::set_symmetric_difference(oneapi::dpl::execution::par_unseq, h_A_keys.begin(), h_A_keys.end(), h_B_keys.begin(), h_B_keys.end(), h_A_vals.begin(), h_B_vals.begin(), h_keys_result.begin(), h_vals_result.begin(), Compare());
  // CHECK-NEXT:   dpct::set_symmetric_difference(oneapi::dpl::execution::par_unseq, h_A_keys.begin(), h_A_keys.end(), h_B_keys.begin(), h_B_keys.end(), h_A_vals.begin(), h_B_vals.begin(), h_keys_result.begin(), h_vals_result.begin(), Compare());
  thrust::set_symmetric_difference_by_key(thrust::host, A_keys, A_keys + 7, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result);
  thrust::set_symmetric_difference_by_key(A_keys, A_keys + 7, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result);
  thrust::set_symmetric_difference_by_key(thrust::host, A_keys, A_keys + 7, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result, Compare());
  thrust::set_symmetric_difference_by_key(A_keys, A_keys + 7, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result, Compare());
  thrust::set_symmetric_difference_by_key(thrust::device, d_A_keys.begin(), d_A_keys.end(), d_B_keys.begin(), d_B_keys.end(), d_A_vals.begin(), d_B_vals.begin(), d_keys_result.begin(), d_vals_result.begin());
  thrust::set_symmetric_difference_by_key(d_A_keys.begin(), d_A_keys.end(), d_B_keys.begin(), d_B_keys.end(), d_A_vals.begin(), d_B_vals.begin(), d_keys_result.begin(), d_vals_result.begin());
  thrust::set_symmetric_difference_by_key(thrust::device, d_A_keys.begin(), d_A_keys.end(), d_B_keys.begin(), d_B_keys.end(), d_A_vals.begin(), d_B_vals.begin(), d_keys_result.begin(), d_vals_result.begin(), Compare());
  thrust::set_symmetric_difference_by_key(d_A_keys.begin(), d_A_keys.end(), d_B_keys.begin(), d_B_keys.end(), d_A_vals.begin(), d_B_vals.begin(), d_keys_result.begin(), d_vals_result.begin(), Compare());
  thrust::set_symmetric_difference_by_key(thrust::host, h_A_keys.begin(), h_A_keys.end(), h_B_keys.begin(), h_B_keys.end(), h_A_vals.begin(), h_B_vals.begin(), h_keys_result.begin(), h_vals_result.begin());
  thrust::set_symmetric_difference_by_key(h_A_keys.begin(), h_A_keys.end(), h_B_keys.begin(), h_B_keys.end(), h_A_vals.begin(), h_B_vals.begin(), h_keys_result.begin(), h_vals_result.begin());
  thrust::set_symmetric_difference_by_key(thrust::host, h_A_keys.begin(), h_A_keys.end(), h_B_keys.begin(), h_B_keys.end(), h_A_vals.begin(), h_B_vals.begin(), h_keys_result.begin(), h_vals_result.begin(), Compare());
  thrust::set_symmetric_difference_by_key(h_A_keys.begin(), h_A_keys.end(), h_B_keys.begin(), h_B_keys.end(), h_A_vals.begin(), h_B_vals.begin(), h_keys_result.begin(), h_vals_result.begin(), Compare());
}

void swap_ranges() {
  thrust::device_vector<int> d_v1(2), d_v2(2);
  thrust::host_vector<int> h_v1(2), h_v2(2);
  int v1[2], v2[2];

  // CHECK:  oneapi::dpl::swap_ranges(oneapi::dpl::execution::make_device_policy(q_ct1), d_v1.begin(), d_v1.end(), d_v2.begin());
  // CHECK-NEXT:  oneapi::dpl::swap_ranges(oneapi::dpl::execution::make_device_policy(q_ct1), d_v1.begin(), d_v1.end(), d_v2.begin());
  // CHECK-NEXT:  oneapi::dpl::swap_ranges(oneapi::dpl::execution::par_unseq, h_v1.begin(), h_v1.end(), h_v2.begin());
  // CHECK-NEXT:  oneapi::dpl::swap_ranges(oneapi::dpl::execution::par_unseq, h_v1.begin(), h_v1.end(), h_v2.begin());
  // CHECK-NEXT:  oneapi::dpl::swap_ranges(oneapi::dpl::execution::par_unseq, v1, v1 + 2, v2);
  // CHECK-NEXT:  oneapi::dpl::swap_ranges(oneapi::dpl::execution::par_unseq, v1, v1 + 2, v2);
  thrust::swap_ranges(thrust::device, d_v1.begin(), d_v1.end(), d_v2.begin());
  thrust::swap_ranges(d_v1.begin(), d_v1.end(), d_v2.begin());
  thrust::swap_ranges(thrust::host, h_v1.begin(), h_v1.end(), h_v2.begin());
  thrust::swap_ranges(h_v1.begin(), h_v1.end(), h_v2.begin());
  thrust::swap_ranges(thrust::host, v1, v1 + 2, v2);
  thrust::swap_ranges(v1, v1 + 2, v2);
}

struct Int {
  __host__ __device__ Int(int x) : val(x) {}
  int val;
};

void uninitialized_fill_n() {

  const int N = 137;
  Int int_val(46);
  int val(46);
  thrust::device_ptr<Int> d_array = thrust::device_malloc<Int>(N);
  int data[N];

  // CHECK:  oneapi::dpl::uninitialized_fill_n(oneapi::dpl::execution::make_device_policy(q_ct1), d_array, N, int_val);
  // CHECK-NEXT:  oneapi::dpl::uninitialized_fill_n(oneapi::dpl::execution::make_device_policy(q_ct1), d_array, N, int_val);
  // CHECK-NEXT:  oneapi::dpl::uninitialized_fill_n(oneapi::dpl::execution::par_unseq, data, N, val);
  // CHECK-NEXT:  oneapi::dpl::uninitialized_fill_n(oneapi::dpl::execution::par_unseq, data, N, val);
  thrust::uninitialized_fill_n(d_array, N, int_val);
  thrust::uninitialized_fill_n(thrust::device, d_array, N, int_val);
  thrust::uninitialized_fill_n(data, N, val);
  thrust::uninitialized_fill_n(thrust::host, data, N, val);
}
