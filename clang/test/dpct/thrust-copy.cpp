// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none --usm-level=none -out-root %T/thrust-copy -in-root=%S %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17 -fsized-deallocation
// RUN: FileCheck --input-file %T/thrust-copy/thrust-copy.cpp.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST  %T/thrust-copy/thrust-copy.cpp.dp.cpp -o %T/thrust-copy/thrust-copy.o.dp.o %}

#ifndef  NO_BUILD_TEST
// CHECK: #include <oneapi/dpl/execution>
// CHECK-NEXT: #include <oneapi/dpl/algorithm>
// CHECK-NEXT: #define DPCT_USM_LEVEL_NONE
// CHECK-NEXT: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <iostream>
// CHECK-NEXT: #include <iterator>
// CHECK-NEXT: #include <dpct/dpl_utils.hpp>

#include <iostream>
#include <iterator>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

// local version of copy.  Make sure it's not migrated
template <typename T>
void copy(T* dst, T* src, int N) {
}

int main(void) {
  // CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
  // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.out_of_order_queue();
  // input data on the host
  const char data[] = "aaabbbbbcddeeeeeeeeeff";
  const size_t N = (sizeof(data) / sizeof(char)) - 1;
  char dst_data[N];
  const char data_d[] = "aaabbbbbcddeeeeeeeeeff";

  // copy input data to the device
  thrust::device_vector<char> input(data_d, data_d + N);
  thrust::host_vector<char> host_input(data, data + N);
  //CHECK: if(dpct::is_device_ptr(data)){
  //CHECK-NEXT:   std::copy(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<const char>(data), dpct::device_pointer<const char>(data + N), dpct::device_pointer<char>(dst_data));
  //CHECK-NEXT: } else {
  //CHECK-NEXT:   std::copy(oneapi::dpl::execution::seq, data, data + N, dst_data);
  //CHECK-NEXT: };
  thrust::copy(data, data + N, dst_data);
  //CHECK: std::copy(oneapi::dpl::execution::seq, host_input.begin(), host_input.end(), std::ostream_iterator<char>(std::cout, ""));
  thrust::copy(host_input.begin(), host_input.end(), std::ostream_iterator<char>(std::cout, ""));
  //CHECK: std::copy(oneapi::dpl::execution::seq, input.begin(), input.end(), std::ostream_iterator<char>(std::cout, ""));
  thrust::copy(input.begin(), input.end(), std::ostream_iterator<char>(std::cout, ""));
  //CHECK: std::copy(oneapi::dpl::execution::seq, host_input.begin(), host_input.end(), std::ostream_iterator<char>(std::cout, ""));
  thrust::copy(thrust::host, host_input.begin(), host_input.end(), std::ostream_iterator<char>(std::cout, ""));
  //CHECK: if(dpct::is_device_ptr(data)){
  //CHECK-NEXT:   std::copy(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<const char>(data), dpct::device_pointer<const char>(data + N), dpct::device_pointer<char>(dst_data));
  //CHECK-NEXT: } else {
  //CHECK-NEXT:   std::copy(oneapi::dpl::execution::seq, data, data + N, dst_data);
  //CHECK-NEXT: };
  thrust::copy(thrust::host, data, data + N, dst_data);
  //CHECK: std::copy(oneapi::dpl::execution::make_device_policy(q_ct1), input.begin(), input.end(), std::ostream_iterator<char>(std::cout, ""));
  thrust::copy(thrust::device, input.begin(), input.end(), std::ostream_iterator<char>(std::cout, ""));

  //CHECK: std::copy_n(oneapi::dpl::execution::make_device_policy(q_ct1), input.begin(), N, std::ostream_iterator<char>(std::cout, ""));
  thrust::copy_n(input.begin(), N, std::ostream_iterator<char>(std::cout, ""));
  //CHECK: std::copy_n(oneapi::dpl::execution::seq, host_input.begin(), N, std::ostream_iterator<char>(std::cout, ""));
  thrust::copy_n(host_input.begin(), N, std::ostream_iterator<char>(std::cout, ""));
  //CHECK: if(dpct::is_device_ptr(data)){
  //CHECK-NEXT:   std::copy_n(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<const char>(data), N, dpct::device_pointer<>(std::ostream_iterator<char>(std::cout, "")));
  //CHECK-NEXT: } else {
  //CHECK-NEXT:   std::copy_n(oneapi::dpl::execution::seq, data, N, std::ostream_iterator<char>(std::cout, ""));
  //CHECK-NEXT: };
  thrust::copy_n(data, N, std::ostream_iterator<char>(std::cout, ""));
  //CHECK: std::copy_n(oneapi::dpl::execution::seq, host_input.begin(), N, std::ostream_iterator<char>(std::cout, ""));
  thrust::copy_n(thrust::host, host_input.begin(), N, std::ostream_iterator<char>(std::cout, ""));
  //CHECK: if(dpct::is_device_ptr(data)){
  //CHECK-NEXT:   std::copy_n(oneapi::dpl::execution::make_device_policy(q_ct1), dpct::device_pointer<const char>(data), N, dpct::device_pointer<char>(dst_data));
  //CHECK-NEXT: } else {
  //CHECK-NEXT:   std::copy_n(oneapi::dpl::execution::seq, data, N, dst_data);
  //CHECK-NEXT: };
  thrust::copy_n(thrust::host, data, N, dst_data);
  //CHECK: std::copy_n(oneapi::dpl::execution::make_device_policy(q_ct1), input.begin(), N, std::ostream_iterator<char>(std::cout, ""));
  thrust::copy_n(thrust::device, input.begin(), N, std::ostream_iterator<char>(std::cout, ""));



  //CHECK:  copy<char>(dst_data, const_cast<char *>(data), N);
  copy<char>(dst_data, const_cast<char *>(data), N);

  std::cout << std::endl << std::endl;
}
#endif
