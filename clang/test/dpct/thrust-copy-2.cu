// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0
// RUN: c2s --sycl-named-lambda --format-range=none --usm-level=none -out-root %T/thrust-copy-2 -in-root=%S %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17 -fsized-deallocation
// RUN: FileCheck --input-file %T/thrust-copy-2/thrust-copy-2.dp.cpp --match-full-lines %s
// CHECK: #include <oneapi/dpl/execution>
// CHECK-NEXT: #include <oneapi/dpl/algorithm>
// CHECK-NEXT: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <c2s/c2s.hpp>
// CHECK-NEXT: #include <c2s/dpl_utils.hpp>

#include <thrust/device_vector.h>
#include <thrust/copy.h>

int main(void) {
// CHECK:   c2s::device_vector<int> input(4, 99);
  thrust::device_vector<int> input(4, 99);
// CHECK:   c2s::device_vector<int> output(4);
  thrust::device_vector<int> output(4);

// CHECK:  std::copy(oneapi::dpl::execution::make_device_policy<class Policy_{{[0-9a-f]+}}>(c2s::get_default_queue()), input.begin(), input.end(), output.begin());
  thrust::copy(input.begin(), input.end(), output.begin());
}

