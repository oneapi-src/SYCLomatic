// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --usm-level=none -out-root %T/thrust-algo-raw-ptr-noneusm-part3 %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17 -fsized-deallocation
// RUN: FileCheck --input-file %T/thrust-algo-raw-ptr-noneusm-part3/thrust-algo-raw-ptr-noneusm-part3.dp.cpp --match-full-lines %s

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

  // CHECK:  if (dpct::is_device_ptr(A)) {
  // CHECK-NEXT:    oneapi::dpl::all_of(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<bool>(A), dpct::device_pointer<bool>(A + 2), oneapi::dpl::identity());
  // CHECK-NEXT:  } else {
  // CHECK-NEXT:    oneapi::dpl::all_of(oneapi::dpl::execution::par_unseq, A, A + 2, oneapi::dpl::identity());
  // CHECK-NEXT:  };
  // CHECK-NEXT:  if (dpct::is_device_ptr(A)) {
  // CHECK-NEXT:    oneapi::dpl::all_of(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<bool>(A), dpct::device_pointer<bool>(A + 2), oneapi::dpl::identity());
  // CHECK-NEXT:  } else {
  // CHECK-NEXT:    oneapi::dpl::all_of(oneapi::dpl::execution::par_unseq, A, A + 2, oneapi::dpl::identity());
  // CHECK-NEXT:  };
  thrust::all_of(thrust::host, A, A + 2, thrust::identity<bool>());
  thrust::all_of(A, A + 2, thrust::identity<bool>());
}

void none_of() {
  bool A[3] = {true, true, false};
  bool result;

  // CHECK:  if (dpct::is_device_ptr(A)) {
  // CHECK-NEXT:    oneapi::dpl::none_of(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<bool>(A), dpct::device_pointer<bool>(A + 2), oneapi::dpl::identity());
  // CHECK-NEXT:  } else {
  // CHECK-NEXT:    oneapi::dpl::none_of(oneapi::dpl::execution::par_unseq, A, A + 2, oneapi::dpl::identity());
  // CHECK-NEXT:  };
  // CHECK-NEXT:  if (dpct::is_device_ptr(A)) {
  // CHECK-NEXT:    oneapi::dpl::none_of(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<bool>(A), dpct::device_pointer<bool>(A + 2), oneapi::dpl::identity());
  // CHECK-NEXT:  } else {
  // CHECK-NEXT:    oneapi::dpl::none_of(oneapi::dpl::execution::par_unseq, A, A + 2, oneapi::dpl::identity());
  // CHECK-NEXT:  };
  thrust::none_of(thrust::host, A, A + 2, thrust::identity<bool>());
  thrust::none_of(A, A + 2, thrust::identity<bool>());
}

struct is_even {
  __host__ __device__ bool operator()(const int &x) const { return (x % 2) == 0; }
};

void is_partitioned() {

  int A[] = {2, 4, 6, 8, 10, 1, 3, 5, 7, 9};
  bool result;

  // CHECK:   if (dpct::is_device_ptr(A)) {
  // CHECK-NEXT:     oneapi::dpl::is_partitioned(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(A), dpct::device_pointer<int>(A + 10), is_even());
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     oneapi::dpl::is_partitioned(oneapi::dpl::execution::par_unseq, A, A + 10, is_even());
  // CHECK-NEXT:   };
  // CHECK-NEXT:   if (dpct::is_device_ptr(A)) {
  // CHECK-NEXT:     oneapi::dpl::is_partitioned(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(A), dpct::device_pointer<int>(A + 10), is_even());
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     oneapi::dpl::is_partitioned(oneapi::dpl::execution::par_unseq, A, A + 10, is_even());
  // CHECK-NEXT:   };
  thrust::is_partitioned(thrust::host, A, A + 10, is_even());
  thrust::is_partitioned(A, A + 10, is_even());
}

void is_sorted_until() {
  int A[8] = {0, 1, 2, 3, 0, 1, 2, 3};
  int *B;
  thrust::greater<int> comp;

  // CHECK:  if (dpct::is_device_ptr(A)) {
  // CHECK-NEXT:    oneapi::dpl::is_sorted_until(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(A), dpct::device_pointer<int>(A + 8));
  // CHECK-NEXT:  } else {
  // CHECK-NEXT:    oneapi::dpl::is_sorted_until(oneapi::dpl::execution::par_unseq, A, A + 8);
  // CHECK-NEXT:  };
  // CHECK-NEXT:  if (dpct::is_device_ptr(A)) {
  // CHECK-NEXT:    oneapi::dpl::is_sorted_until(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(A), dpct::device_pointer<int>(A + 8));
  // CHECK-NEXT:  } else {
  // CHECK-NEXT:    oneapi::dpl::is_sorted_until(oneapi::dpl::execution::par_unseq, A, A + 8);
  // CHECK-NEXT:  };
  // CHECK-NEXT:  if (dpct::is_device_ptr(A)) {
  // CHECK-NEXT:    oneapi::dpl::is_sorted_until(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(A), dpct::device_pointer<int>(A + 8), comp);
  // CHECK-NEXT:  } else {
  // CHECK-NEXT:    oneapi::dpl::is_sorted_until(oneapi::dpl::execution::par_unseq, A, A + 8, comp);
  // CHECK-NEXT:  };
  // CHECK-NEXT:  if (dpct::is_device_ptr(A)) {
  // CHECK-NEXT:    oneapi::dpl::is_sorted_until(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(A), dpct::device_pointer<int>(A + 8), comp);
  // CHECK-NEXT:  } else {
  // CHECK-NEXT:    oneapi::dpl::is_sorted_until(oneapi::dpl::execution::par_unseq, A, A + 8, comp);
  // CHECK-NEXT:  };
  thrust::is_sorted_until(thrust::host, A, A + 8);
  thrust::is_sorted_until(A, A + 8);
  thrust::is_sorted_until(thrust::host, A, A + 8, comp);
  thrust::is_sorted_until(A, A + 8, comp);
}

void set_intersection() {
  int A1[6] = {1, 3, 5, 7, 9, 11};
  int A2[7] = {1, 1, 2, 3, 5, 8, 13};
  int result[3];
  int *result_end;

  // CHECK:  if (dpct::is_device_ptr(A1)) {
  // CHECK-NEXT:    oneapi::dpl::set_intersection(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(A1), dpct::device_pointer<int>(A1 + 6), dpct::device_pointer<int>(A2), dpct::device_pointer<int>(A2 + 7), dpct::device_pointer<int>(result));
  // CHECK-NEXT:  } else {
  // CHECK-NEXT:    oneapi::dpl::set_intersection(oneapi::dpl::execution::par_unseq, A1, A1 + 6, A2, A2 + 7, result);
  // CHECK-NEXT:  };
  // CHECK-NEXT:  if (dpct::is_device_ptr(A1)) {
  // CHECK-NEXT:    oneapi::dpl::set_intersection(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(A1), dpct::device_pointer<int>(A1 + 6), dpct::device_pointer<int>(A2), dpct::device_pointer<int>(A2 + 7), dpct::device_pointer<int>(result));
  // CHECK-NEXT:  } else {
  // CHECK-NEXT:    oneapi::dpl::set_intersection(oneapi::dpl::execution::par_unseq, A1, A1 + 6, A2, A2 + 7, result);
  // CHECK-NEXT:  };
  // CHECK-NEXT:  if (dpct::is_device_ptr(A1)) {
  // CHECK-NEXT:    oneapi::dpl::set_intersection(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(A1), dpct::device_pointer<int>(A1 + 6), dpct::device_pointer<int>(A2), dpct::device_pointer<int>(A2 + 7), dpct::device_pointer<int>(result), std::greater<int>());
  // CHECK-NEXT:  } else {
  // CHECK-NEXT:    oneapi::dpl::set_intersection(oneapi::dpl::execution::par_unseq, A1, A1 + 6, A2, A2 + 7, result, std::greater<int>());
  // CHECK-NEXT:  };
  // CHECK-NEXT:  if (dpct::is_device_ptr(A1)) {
  // CHECK-NEXT:    oneapi::dpl::set_intersection(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<int>(A1), dpct::device_pointer<int>(A1 + 6), dpct::device_pointer<int>(A2), dpct::device_pointer<int>(A2 + 7), dpct::device_pointer<int>(result), std::greater<int>());
  // CHECK-NEXT:  } else {
  // CHECK-NEXT:    oneapi::dpl::set_intersection(oneapi::dpl::execution::par_unseq, A1, A1 + 6, A2, A2 + 7, result, std::greater<int>());
  // CHECK-NEXT:  };
  thrust::set_intersection(thrust::host, A1, A1 + 6, A2, A2 + 7, result);
  thrust::set_intersection(A1, A1 + 6, A2, A2 + 7, result);
  thrust::set_intersection(thrust::host, A1, A1 + 6, A2, A2 + 7, result, thrust::greater<int>());
  thrust::set_intersection(A1, A1 + 6, A2, A2 + 7, result, thrust::greater<int>());
}
