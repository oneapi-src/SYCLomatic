// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -out-root %T/thrust-algo-part3 %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/thrust-algo-part3/thrust-algo-part3.dp.cpp --match-full-lines %s

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>

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
