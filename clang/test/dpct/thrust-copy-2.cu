// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0
// RUN: dpct --sycl-named-lambda --format-range=none --usm-level=none -out-root %T -in-root=%S %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/thrust-copy-2.dp.cpp --match-full-lines %s
// CHECK: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <dpct/dpstd_utils.hpp>
// CHECK-NEXT: #include <dpstd/execution>
// CHECK-NEXT: #include <dpstd/algorithm>

#include <thrust/device_vector.h>
#include <thrust/copy.h>

int main(void) {
// CHECK:   dpct::device_vector<int> input(4, 99);
  thrust::device_vector<int> input(4, 99);
// CHECK:   dpct::device_vector<int> output(4);
  thrust::device_vector<int> output(4);

// CHECK:  std::copy(dpstd::execution::make_device_policy<class Policy_{{[0-9a-f]+}}>(dpct::get_default_queue()), input.begin(), input.end(), output.begin());
  thrust::copy(input.begin(), input.end(), output.begin());
}
