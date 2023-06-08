// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --usm-level=none -out-root %T/thrust-algo-raw-ptr-noneusm-part3 %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17 -fsized-deallocation
// RUN: FileCheck --input-file %T/thrust-algo-raw-ptr-noneusm-part3/thrust-algo-raw-ptr-noneusm-part3.dp.cpp --match-full-lines %s

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>

void all_of() {

  bool A[3] = {true, true, false};
  bool result;

  // CHECK:  if (dpct::is_device_ptr(A)) {
  // CHECK-NEXT:    oneapi::dpl::all_of(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<bool>(A), dpct::device_pointer<bool>(A + 2), oneapi::dpl::identity());
  // CHECK-NEXT:  } else {
  // CHECK-NEXT:    oneapi::dpl::all_of(oneapi::dpl::execution::seq, A, A + 2, oneapi::dpl::identity());
  // CHECK-NEXT:  };
  // CHECK-NEXT:  if (dpct::is_device_ptr(A)) {
  // CHECK-NEXT:    oneapi::dpl::all_of(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<bool>(A), dpct::device_pointer<bool>(A + 2), oneapi::dpl::identity());
  // CHECK-NEXT:  } else {
  // CHECK-NEXT:    oneapi::dpl::all_of(oneapi::dpl::execution::seq, A, A + 2, oneapi::dpl::identity());
  // CHECK-NEXT:  };
  thrust::all_of(thrust::host, A, A + 2, thrust::identity<bool>());
  thrust::all_of(A, A + 2, thrust::identity<bool>());
}
