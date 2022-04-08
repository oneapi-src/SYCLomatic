// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: c2s --format-range=none -out-root %T/thrust-reduce %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -fno-delayed-template-parsing -std=c++17 -fsized-deallocation
// RUN: FileCheck --input-file %T/thrust-reduce/thrust-reduce.dp.cpp --match-full-lines %s
// CHECK: #include <oneapi/dpl/execution>
// CHECK-NEXT: #include <oneapi/dpl/algorithm>
// CHECK-NEXT: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <c2s/c2s.hpp>
// CHECK-NEXT: #include <c2s/dpl_utils.hpp>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>

template <typename T>
struct square {
// CHECK:  T operator()(const T& x) const { return x * x; }
  __host__ __device__  T operator()(const T& x) const { return x * x; }
};

int main() {
  double sum;
  double *p;
// CHECK:  c2s::device_pointer<double> dp(p);
  thrust::device_ptr<double> dp(p);
// CHECK:  sum = std::reduce(oneapi::dpl::execution::make_device_policy(c2s::get_default_queue()), dp, dp + 10);
  sum = thrust::reduce(dp, dp + 10);
}

void check_transform_reduce() {
// CHECK:  c2s::device_ext &dev_ct1 = c2s::get_current_device();
// CHECK-NEXT:  sycl::queue &q_ct1 = dev_ct1.default_queue();
  float x[4] = {1.0, 2.0, 3.0, 4.0};
// CHECK:  c2s::device_vector<float> d_x(x, x + 4);
  thrust::device_vector<float> d_x(x, x + 4);
  square<float>        unary_op;
// CHECK:  std::plus<float> binary_op;
  thrust::plus<float> binary_op;
  float init = 0;

// CHECK:  float norm     = std::transform_reduce(oneapi::dpl::execution::make_device_policy(q_ct1), d_x.begin(), d_x.end(), init, binary_op, unary_op);
  float norm     = thrust::transform_reduce(d_x.begin(), d_x.end(), unary_op, init, binary_op);
// CHECK:  float normSqrt = std::sqrt(std::transform_reduce(oneapi::dpl::execution::make_device_policy(q_ct1), d_x.begin(), d_x.end(), init, binary_op, unary_op));
  float normSqrt = std::sqrt(thrust::transform_reduce(d_x.begin(), d_x.end(), unary_op, init, binary_op));
}

template <typename T>
class C {
  T *data;
public:
  C() {
    this->data = 0;
  }

  // CHECK:   inline T *raw() {
  // CHECK-NEXT:   return c2s::get_raw_pointer(this->data);
  // CHECK-NEXT: }
  // CHECK-NEXT: inline const T *raw() const {
  // CHECK-NEXT:   return c2s::get_raw_pointer(this->data + 2);
  // CHECK-NEXT: }
  inline T *raw() {
    return thrust::raw_pointer_cast(this->data);
  }
  inline const T *raw() const {
    return thrust::raw_pointer_cast(this->data + 2);
  }
};

