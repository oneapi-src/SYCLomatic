// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/thrust-reduce.dp.cpp --match-full-lines %s
// CHECK: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <dpct/dpstd_utils.hpp>
// CHECK-NEXT: #include <dpstd/execution>
// CHECK-NEXT: #include <dpstd/algorithm>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

int main() {
  double sum;
  double *p;
// CHECK:  dpct::device_ptr<double> dp(p);
  thrust::device_ptr<double> dp(p);
// CHECK:  sum = std::reduce(dpstd::execution::make_device_policy(dpct::get_default_queue()), dp, dp + 10);
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
  // CHECK-NEXT:   return dpct::raw_pointer_cast(this->data);
  // CHECK-NEXT: }
  // CHECK-NEXT: inline const T *raw() const {
  // CHECK-NEXT:   return dpct::raw_pointer_cast(this->data + 2);
  // CHECK-NEXT: }
  inline T *raw() {
    return thrust::raw_pointer_cast(this->data);
  }
  inline const T *raw() const {
    return thrust::raw_pointer_cast(this->data + 2);
  }
};
