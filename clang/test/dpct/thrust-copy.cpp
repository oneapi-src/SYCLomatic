// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none --usm-level=none -out-root %T/thrust-copy -in-root=%S %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17 -fsized-deallocation
// RUN: FileCheck --input-file %T/thrust-copy/thrust-copy.cpp.dp.cpp --match-full-lines %s
// CHECK: #include <oneapi/dpl/execution>
// CHECK-NEXT: #include <oneapi/dpl/algorithm>
// CHECK-NEXT: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <iostream>
// CHECK-NEXT: #include <iterator>
// CHECK-NEXT: #include <dpct/dpl_utils.hpp>

#include <iostream>
#include <iterator>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

// local version of copy.  Make sure it's not migrated
template <typename T>
void copy(T* dst, T* src, int N) {
}

int main(void) {
  // CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
  // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.default_queue();
  // input data on the host
  const char data[] = "aaabbbbbcddeeeeeeeeeff";
  const size_t N = (sizeof(data) / sizeof(char)) - 1;
  char dst_data[N];

  // copy input data to the device
// CHECK:   dpct::device_vector<char> input(data, data + N);
  thrust::device_vector<char> input(data, data + N);

  std::cout << "input data:" << std::endl;
// CHECK:  std::copy(oneapi::dpl::execution::make_device_policy(q_ct1), input.begin(), input.end(), std::ostream_iterator<char>(std::cout, ""));
  thrust::copy(input.begin(), input.end(), std::ostream_iterator<char>(std::cout, ""));
// CHECK:  std::copy_n(oneapi::dpl::execution::make_device_policy(q_ct1), input.begin(), N, std::ostream_iterator<char>(std::cout, ""));
  thrust::copy_n(input.begin(), N, std::ostream_iterator<char>(std::cout, ""));
// CHECK:  copy<char>(dst_data, const_cast<char *>(data), N);
  copy<char>(dst_data, const_cast<char *>(data), N);

  std::cout << std::endl << std::endl;
}
