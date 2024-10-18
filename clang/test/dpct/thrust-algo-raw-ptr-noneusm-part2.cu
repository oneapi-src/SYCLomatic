// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --usm-level=none -out-root %T/thrust-algo-raw-ptr-noneusm-part2 %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17 -fsized-deallocation
// RUN: FileCheck --input-file %T/thrust-algo-raw-ptr-noneusm-part2/thrust-algo-raw-ptr-noneusm-part2.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST  %T/thrust-algo-raw-ptr-noneusm-part2/thrust-algo-raw-ptr-noneusm-part2.dp.cpp -o %T/thrust-algo-raw-ptr-noneusm-part2/thrust-algo-raw-ptr-noneusm-part2.dp.o %}

#ifndef NO_BUILD_TEST
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

void set_symmetric_difference_by_key() {
  int A_keys[7] = {0, 1, 2, 2, 4, 6, 7};
  int A_vals[7] = {0, 0, 0, 0, 0, 0, 0};
  int B_keys[5] = {1, 1, 2, 5, 8};
  int B_vals[5] = {1, 1, 1, 1, 1};
  int keys_result[8];
  int vals_result[8];

  // CHECK:  if (dpct::is_device_ptr(A_keys)) {
  // CHECK-NEXT:    dpct::set_symmetric_difference(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(A_keys), dpct::device_pointer<int>(A_keys + 7), dpct::device_pointer<int>(B_keys), dpct::device_pointer<int>(B_keys + 5), dpct::device_pointer<int>(A_vals), dpct::device_pointer<int>(B_vals), dpct::device_pointer<int>(keys_result), dpct::device_pointer<int>(vals_result));
  // CHECK-NEXT:  } else {
  // CHECK-NEXT:    dpct::set_symmetric_difference(oneapi::dpl::execution::seq, A_keys, A_keys + 7, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result);
  // CHECK-NEXT:  };
  // CHECK-NEXT:  if (dpct::is_device_ptr(A_keys)) {
  // CHECK-NEXT:    dpct::set_symmetric_difference(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(A_keys), dpct::device_pointer<int>(A_keys + 7), dpct::device_pointer<int>(B_keys), dpct::device_pointer<int>(B_keys + 5), dpct::device_pointer<int>(A_vals), dpct::device_pointer<int>(B_vals), dpct::device_pointer<int>(keys_result), dpct::device_pointer<int>(vals_result));
  // CHECK-NEXT:  } else {
  // CHECK-NEXT:    dpct::set_symmetric_difference(oneapi::dpl::execution::seq, A_keys, A_keys + 7, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result);
  // CHECK-NEXT:  };
  // CHECK-NEXT:  if (dpct::is_device_ptr(A_keys)) {
  // CHECK-NEXT:    dpct::set_symmetric_difference(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(A_keys), dpct::device_pointer<int>(A_keys + 7), dpct::device_pointer<int>(B_keys), dpct::device_pointer<int>(B_keys + 5), dpct::device_pointer<int>(A_vals), dpct::device_pointer<int>(B_vals), dpct::device_pointer<int>(keys_result), dpct::device_pointer<int>(vals_result), Compare());
  // CHECK-NEXT:  } else {
  // CHECK-NEXT:    dpct::set_symmetric_difference(oneapi::dpl::execution::seq, A_keys, A_keys + 7, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result, Compare());
  // CHECK-NEXT:  };
  // CHECK-NEXT:  if (dpct::is_device_ptr(A_keys)) {
  // CHECK-NEXT:    dpct::set_symmetric_difference(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(A_keys), dpct::device_pointer<int>(A_keys + 7), dpct::device_pointer<int>(B_keys), dpct::device_pointer<int>(B_keys + 5), dpct::device_pointer<int>(A_vals), dpct::device_pointer<int>(B_vals), dpct::device_pointer<int>(keys_result), dpct::device_pointer<int>(vals_result), Compare());
  // CHECK-NEXT:  } else {
  // CHECK-NEXT:    dpct::set_symmetric_difference(oneapi::dpl::execution::seq, A_keys, A_keys + 7, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result, Compare());
  // CHECK-NEXT:  };
  thrust::set_symmetric_difference_by_key(thrust::host, A_keys, A_keys + 7, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result);
  thrust::set_symmetric_difference_by_key(A_keys, A_keys + 7, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result);
  thrust::set_symmetric_difference_by_key(thrust::host, A_keys, A_keys + 7, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result, Compare());
  thrust::set_symmetric_difference_by_key(A_keys, A_keys + 7, B_keys, B_keys + 5, A_vals, B_vals, keys_result, vals_result, Compare());
}

void swap_ranges() {
  int v1[2], v2[2];

  // CHECK:  if (dpct::is_device_ptr(v1)) {
  // CHECK-NEXT:    oneapi::dpl::swap_ranges(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(v1), dpct::device_pointer<int>(v1 + 2), dpct::device_pointer<int>(v2));
  // CHECK-NEXT:  } else {
  // CHECK-NEXT:    oneapi::dpl::swap_ranges(oneapi::dpl::execution::seq, v1, v1 + 2, v2);
  // CHECK-NEXT:  };
  // CHECK-NEXT:  if (dpct::is_device_ptr(v1)) {
  // CHECK-NEXT:    oneapi::dpl::swap_ranges(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(v1), dpct::device_pointer<int>(v1 + 2), dpct::device_pointer<int>(v2));
  // CHECK-NEXT:  } else {
  // CHECK-NEXT:    oneapi::dpl::swap_ranges(oneapi::dpl::execution::seq, v1, v1 + 2, v2);
  // CHECK-NEXT:  };
  thrust::swap_ranges(thrust::host, v1, v1 + 2, v2);
  thrust::swap_ranges(v1, v1 + 2, v2);
}

struct Int {
  __host__ __device__ Int(int x) : val(x) {}
  int val;
};

void uninitialized_fill_n() {

  const int N = 137;
  int val(46);
  int data[N];

  // CHECK:  if (dpct::is_device_ptr(data)) {
  // CHECK-NEXT:    oneapi::dpl::uninitialized_fill_n(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(data), N, val);
  // CHECK-NEXT:  } else {
  // CHECK-NEXT:    oneapi::dpl::uninitialized_fill_n(oneapi::dpl::execution::seq, data, N, val);
  // CHECK-NEXT:  };
  // CHECK-NEXT:  if (dpct::is_device_ptr(data)) {
  // CHECK-NEXT:    oneapi::dpl::uninitialized_fill_n(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(data), N, val);
  // CHECK-NEXT:  } else {
  // CHECK-NEXT:    oneapi::dpl::uninitialized_fill_n(oneapi::dpl::execution::seq, data, N, val);
  // CHECK-NEXT:  };
  thrust::uninitialized_fill_n(data, N, val);
  thrust::uninitialized_fill_n(thrust::host, data, N, val);
}

#endif
