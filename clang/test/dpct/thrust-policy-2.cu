// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0
// RUN: dpct --format-range=none -out-root %T/thrust-policy-2 %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++14
// RUN: FileCheck %s --match-full-lines --input-file %T/thrust-policy-2/thrust-policy-2.dp.cpp

#include <thrust/execution_policy.h>

class MyAlloctor {};
void foo(cudaStream_t stream) {
  MyAlloctor thrust_allocator;
  // CHECK: auto p = oneapi::dpl::execution::make_device_policy(*stream);
  auto p = thrust::cuda::par(thrust_allocator).on(stream);
}
