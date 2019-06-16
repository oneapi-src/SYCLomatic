// UNSUPPORTED: cuda-8.0
// RUN: syclct -out-root %T %s  -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck --input-file %T/thrust-copy.cc_sycl.cpp --match-full-lines %s
// CHECK: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <syclct/syclct.hpp>
// CHECK-NEXT: #include <iostream>
// CHECK-NEXT: #include <iterator>
// CHECK-NEXT: #include <dpstd/containers>
// CHECK-NEXT: #include <dpstd/algorithm>
// CHECK-NEXT: #include <dpstd/execution>
#include <iostream>
#include <iterator>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

int main(void) {
  // input data on the host
  const char data[] = "aaabbbbbcddeeeeeeeeeff";
  const size_t N = (sizeof(data) / sizeof(char)) - 1;

  // copy input data to the device
// CHECK:   dpstd::device_vector<char> input(data, data + N);
  thrust::device_vector<char> input(data, data + N);

  std::cout << "input data:" << std::endl;
// CHECK:  std::copy(dpstd::execution::sycl, input.begin(), input.end(), std::ostream_iterator<char>(std::cout, ""));
  thrust::copy(input.begin(), input.end(), std::ostream_iterator<char>(std::cout, ""));
// CHECK:  std::copy_n(dpstd::execution::sycl, input.begin(), N, std::ostream_iterator<char>(std::cout, ""));
  thrust::copy_n(input.begin(), N, std::ostream_iterator<char>(std::cout, ""));
  std::cout << std::endl << std::endl;
}
