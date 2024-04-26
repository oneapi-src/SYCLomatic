// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.2, cuda-11.8
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.2, v11.8
// RUN: dpct --format-range=none -out-root %T/thrust-algo-part4 %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/thrust-algo-part4/thrust-algo-part4.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/thrust-algo-part4/thrust-algo-part4.dp.cpp -o %T/thrust-algo-part4/thrust-algo-part4.dp.o %}

#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>
#include <thrust/partition.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

void unique_count() {
  const int N = 7;
  int count;
  int A[N] = {1, 3, 3, 3, 2, 2, 1};
  thrust::host_vector<int> h_A(A, A + N);
  thrust::device_vector<int> d_A(A, A + N);

  // CHECK:  count = dpct::unique_count(oneapi::dpl::execution::seq, A, A + N, oneapi::dpl::equal_to<int>());
  // CHECK-NEXT:  count = dpct::unique_count(oneapi::dpl::execution::seq, A, A + N, oneapi::dpl::equal_to<int>());
  // CHECK-NEXT:  count = dpct::unique_count(oneapi::dpl::execution::seq, A, A + N);
  // CHECK-NEXT:  count = dpct::unique_count(oneapi::dpl::execution::seq, A, A + N);
  // CHECK-NEXT:  count = dpct::unique_count(oneapi::dpl::execution::seq, h_A.begin(), h_A.begin() + N, oneapi::dpl::equal_to<int>());
  // CHECK-NEXT:  count = dpct::unique_count(oneapi::dpl::execution::make_device_policy(q_ct1), d_A.begin(), d_A.begin() + N, oneapi::dpl::equal_to<int>());
  // CHECK-NEXT:  count = dpct::unique_count(oneapi::dpl::execution::seq, h_A.begin(), h_A.begin() + N, oneapi::dpl::equal_to<int>());
  // CHECK-NEXT:  count = dpct::unique_count(oneapi::dpl::execution::make_device_policy(q_ct1), d_A.begin(), d_A.begin() + N, oneapi::dpl::equal_to<int>());
  // CHECK-NEXT:  count = dpct::unique_count(oneapi::dpl::execution::seq, h_A.begin(), h_A.begin() + N);
  // CHECK-NEXT:  count = dpct::unique_count(oneapi::dpl::execution::make_device_policy(q_ct1), d_A.begin(), d_A.begin() + N);
  // CHECK-NEXT:  count = dpct::unique_count(oneapi::dpl::execution::seq, h_A.begin(), h_A.begin() + N);
  // CHECK-NEXT:  count = dpct::unique_count(oneapi::dpl::execution::make_device_policy(q_ct1), d_A.begin(), d_A.begin() + N);
  count = thrust::unique_count(thrust::host, A, A + N, thrust::equal_to<int>());
  count = thrust::unique_count(A, A + N, thrust::equal_to<int>());
  count = thrust::unique_count(thrust::host, A, A + N);
  count = thrust::unique_count(A, A + N);
  count = thrust::unique_count(thrust::host, h_A.begin(), h_A.begin() + N, thrust::equal_to<int>());
  count = thrust::unique_count(thrust::device, d_A.begin(), d_A.begin() + N, thrust::equal_to<int>());
  count = thrust::unique_count(h_A.begin(), h_A.begin() + N, thrust::equal_to<int>());
  count = thrust::unique_count(d_A.begin(), d_A.begin() + N, thrust::equal_to<int>());
  count = thrust::unique_count(thrust::host, h_A.begin(), h_A.begin() + N);
  count = thrust::unique_count(thrust::device, d_A.begin(), d_A.begin() + N);
  count = thrust::unique_count(h_A.begin(), h_A.begin() + N);
  count = thrust::unique_count(d_A.begin(), d_A.begin() + N);
}
