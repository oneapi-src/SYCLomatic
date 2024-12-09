// UNSUPPORTED: cuda-8.0, cuda-9.2, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.2, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/thrust_system_error %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/thrust_system_error/thrust_system_error.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST  %T/thrust_system_error/thrust_system_error.dp.cpp -o %T/thrust_system_error/thrust_system_error.dp.o %}

#include <cuda_runtime_api.h>
#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>
#include <string>

void cuda_safe_call(cudaError_t error, const std::string &message = "") {

  // CHECK: std::errc e;
  thrust::errc::errc_t e;

  // CHECK: std::system_error(error, std::generic_category(), message);
  std::system_error(error, std::generic_category(), message);

  // CHECK:	std::make_error_condition(e);
  // CHECK-NEXT:	std::make_error_condition(e);
  // CHECK-NEXT:	std::generic_category();
  // CHECK-NEXT:  std::generic_category();
  // CHECK-NEXT:	std::generic_category();
  // CHECK-NEXT:	std::generic_category();
  // CHECK-NEXT:	std::system_category();
  // CHECK-NEXT:	std::system_category();
  // CHECK-NEXT:	std::error_code t1(static_cast<int>(0), std::generic_category());
  // CHECK-NEXT:	std::error_code t2(static_cast<int>(0), std::generic_category());
  // CHECK-NEXT:  std::error_code t3(static_cast<int>(0), std::generic_category());
  // CHECK-NEXT:  std::error_code t4(static_cast<int>(0), std::generic_category());
  // CHECK-NEXT:  std::error_condition t5(0, std::generic_category());
  // CHECK-NEXT:  std::error_condition t6(0, std::generic_category());
  thrust::make_error_condition(e);
  thrust::system::make_error_condition(e);
  thrust::cuda_category();
  thrust::system::cuda_category();
  thrust::generic_category();
  thrust::system::generic_category();
  thrust::system_category();
  thrust::system::system_category();
  thrust::error_code t1(static_cast<int>(0), thrust::generic_category());
  thrust::system::error_code t2(static_cast<int>(0), thrust::generic_category());
  thrust::error_code t3(static_cast<int>(0), thrust::generic_category());
  thrust::system::error_code t4(static_cast<int>(0), thrust::generic_category());
  thrust::error_condition t5(0, thrust::generic_category());
  thrust::system::error_condition t6(0, thrust::generic_category());

  if (error) {
    // CHECK: throw std::system_error(error, std::generic_category(), message);
    throw thrust::system_error(error, thrust::cuda_category(), message);
  }
}

void foo() {
  thrust::error_code t1(static_cast<int>(0), thrust::generic_category());
  std::string arg;
  char *char_arg = "test";
  int ev = 0;
  // CHECK: std::system_error test_1(t1, arg);
  // CHECK-NEXT: std::system_error test_2(t1, char_arg);
  // CHECK-NEXT: std::system_error test_3(t1);
  // CHECK-NEXT: std::system_error test_4(ev, std::generic_category(), arg);
  // CHECK-NEXT: std::system_error test_5(ev, std::generic_category(), char_arg);
  // CHECK-NEXT: std::system_error test_6(ev, std::generic_category());
  // CHECK-NEXT: std::system_error test_7(t1, arg);
  // CHECK-NEXT: std::system_error test_8(t1, char_arg);
  // CHECK-NEXT: std::system_error test_9(t1);
  // CHECK-NEXT: std::system_error test_10(ev, std::generic_category(), arg);
  // CHECK-NEXT: std::system_error test_11(ev, std::generic_category(), char_arg);
  // CHECK-NEXT: std::system_error test_12(ev, std::generic_category());
  thrust::system::system_error test_1(t1, arg);
  thrust::system::system_error test_2(t1, char_arg);
  thrust::system::system_error test_3(t1);
  thrust::system::system_error test_4(ev, thrust::cuda_category(), arg);
  thrust::system::system_error test_5(ev, thrust::cuda_category(), char_arg);
  thrust::system::system_error test_6(ev, thrust::cuda_category());
  thrust::system_error test_7(t1, arg);
  thrust::system_error test_8(t1, char_arg);
  thrust::system_error test_9(t1);
  thrust::system_error test_10(ev, thrust::cuda_category(), arg);
  thrust::system_error test_11(ev, thrust::cuda_category(), char_arg);
  thrust::system_error test_12(ev, thrust::cuda_category());
}

int main() {
// CHECK: dpct::err0 e = 1;  
  cudaError_t e = cudaErrorInvalidValue;  
  cuda_safe_call(e);
  return 0;
}
