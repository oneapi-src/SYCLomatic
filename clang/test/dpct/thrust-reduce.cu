// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -fno-delayed-template-parsing
// RUN: FileCheck --input-file %T/thrust-reduce.dp.cpp --match-full-lines %s
// CHECK: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <dpct/dpl_utils.hpp>
// CHECK-NEXT: #include <oneapi/dpl/execution>
// CHECK-NEXT: #include <oneapi/dpl/algorithm>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

int main() {
  double sum;
  double *p;
// CHECK:  dpct::device_pointer<double> dp(p);
  thrust::device_ptr<double> dp(p);
// CHECK:  sum = std::reduce(oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()), dp, dp + 10);
  sum = thrust::reduce(dp, dp + 10);
}

template <typename T>
class C {
  T *data;
public:
  C() {
    this->data = 0;
  }

  // CHECK:   inline T *raw() {
  // CHECK-NEXT:   return dpct::get_raw_pointer(this->data);
  // CHECK-NEXT: }
  // CHECK-NEXT: inline const T *raw() const {
  // CHECK-NEXT:   return dpct::get_raw_pointer(this->data + 2);
  // CHECK-NEXT: }
  inline T *raw() {
    return thrust::raw_pointer_cast(this->data);
  }
  inline const T *raw() const {
    return thrust::raw_pointer_cast(this->data + 2);
  }
};
