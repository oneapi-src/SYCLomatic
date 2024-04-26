// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -out-root %T/thrust-reduce %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17 -fsized-deallocation -ferror-limit=50
// RUN: FileCheck --input-file %T/thrust-reduce/thrust-reduce.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/thrust-reduce/thrust-reduce.dp.cpp -o %T/thrust-reduce/thrust-reduce.dp.o %}
// CHECK: #include <oneapi/dpl/execution>
// CHECK-NEXT: #include <oneapi/dpl/algorithm>
// CHECK-NEXT: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <dpct/dpl_utils.hpp>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>

template <typename T>
struct square {
// CHECK:  T operator()(const T& x) const { return x * x; }
  __host__ __device__  T operator()(const T& x) const { return x * x; }
};

int main() {
  double sum;
  double *p;
// CHECK:  dpct::device_pointer<double> dp(p);
  thrust::device_ptr<double> dp(p);
// CHECK:  sum = std::reduce(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), dp, dp + 10);
  sum = thrust::reduce(dp, dp + 10);
}

void check_transform_reduce() {
// CHECK:  dpct::device_ext &dev_ct1 = dpct::get_current_device();
// CHECK-NEXT:  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  float x[4] = {1.0, 2.0, 3.0, 4.0};
// CHECK:  dpct::device_vector<float> d_x(x, x + 4);
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

class MyAlloctor {
public:
  typedef char value_type;
  char *allocate(std::ptrdiff_t size) { return nullptr; }
  void deallocate(char *p, size_t size) {}
};

void thrust_test() {
  int data[6] = {1, 0, 2, 2, 1, 3};
  thrust::device_vector<int> d_data(data, data + 6);
  thrust::host_vector<int> h_data(data, data + 6);
  int result;
  MyAlloctor thrust_allocator;

  // CHECK:  result = std::reduce(oneapi::dpl::execution::seq, data, data + 6);
  // CHECK-NEXT:  result = std::reduce(oneapi::dpl::execution::seq, data, data + 6);
  // CHECK-NEXT:  result = std::reduce(oneapi::dpl::execution::seq, data, data + 6, 1);
  // CHECK-NEXT:  result = std::reduce(oneapi::dpl::execution::seq, data, data + 6, 1);
  // CHECK-NEXT:  result = std::reduce(oneapi::dpl::execution::seq, data, data + 6, -1, oneapi::dpl::maximum<int>());
  // CHECK-NEXT:  result = std::reduce(oneapi::dpl::execution::seq, data, data + 6, -1, oneapi::dpl::maximum<int>());
  // CHECK-NEXT:  result = std::reduce(oneapi::dpl::execution::make_device_policy(q_ct1), d_data.begin(), d_data.begin() + 6);
  // CHECK-NEXT:  result = std::reduce(oneapi::dpl::execution::make_device_policy(q_ct1), d_data.begin(), d_data.begin() + 6);
  // CHECK-NEXT:  result = std::reduce(oneapi::dpl::execution::make_device_policy(q_ct1), d_data.begin(), d_data.begin() + 6, 1);
  // CHECK-NEXT:  result = std::reduce(oneapi::dpl::execution::make_device_policy(q_ct1), d_data.begin(), d_data.begin() + 6, 1);
  // CHECK-NEXT:  result = std::reduce(oneapi::dpl::execution::make_device_policy(q_ct1), d_data.begin(), d_data.begin() + 6, -1, oneapi::dpl::maximum<int>());
  // CHECK-NEXT:  result = std::reduce(oneapi::dpl::execution::make_device_policy(q_ct1), d_data.begin(), d_data.begin() + 6, -1, oneapi::dpl::maximum<int>());
  // CHECK-NEXT:  result = std::reduce(oneapi::dpl::execution::seq, h_data.begin(), h_data.begin() + 6);
  // CHECK-NEXT:  result = std::reduce(oneapi::dpl::execution::seq, h_data.begin(), h_data.begin() + 6);
  // CHECK-NEXT:  result = std::reduce(oneapi::dpl::execution::seq, h_data.begin(), h_data.begin() + 6, 1);
  // CHECK-NEXT:  result = std::reduce(oneapi::dpl::execution::seq, h_data.begin(), h_data.begin() + 6, 1);
  // CHECK-NEXT:  result = std::reduce(oneapi::dpl::execution::seq, h_data.begin(), h_data.begin() + 6, -1, oneapi::dpl::maximum<int>());
  // CHECK-NEXT:  result = std::reduce(oneapi::dpl::execution::seq, h_data.begin(), h_data.begin() + 6, -1, oneapi::dpl::maximum<int>());
  // CHECK-NEXT:  result = std::reduce(oneapi::dpl::execution::make_device_policy(q_ct1), d_data.begin(), d_data.begin() + 6, -1, oneapi::dpl::maximum<int>());
  result = thrust::reduce(thrust::host, data, data + 6);
  result = thrust::reduce(data, data + 6);
  result = thrust::reduce(thrust::host, data, data + 6, 1);
  result = thrust::reduce(data, data + 6, 1);
  result = thrust::reduce(thrust::host, data, data + 6, -1, thrust::maximum<int>());
  result = thrust::reduce(data, data + 6, -1, thrust::maximum<int>());
  result = thrust::reduce(thrust::device, d_data.begin(), d_data.begin() + 6);
  result = thrust::reduce(d_data.begin(), d_data.begin() + 6);
  result = thrust::reduce(thrust::device, d_data.begin(), d_data.begin() + 6, 1);
  result = thrust::reduce(d_data.begin(), d_data.begin() + 6, 1);
  result = thrust::reduce(thrust::device, d_data.begin(), d_data.begin() + 6, -1, thrust::maximum<int>());
  result = thrust::reduce(d_data.begin(), d_data.begin() + 6, -1, thrust::maximum<int>());
  result = thrust::reduce(thrust::host, h_data.begin(), h_data.begin() + 6);
  result = thrust::reduce(h_data.begin(), h_data.begin() + 6);
  result = thrust::reduce(thrust::host, h_data.begin(), h_data.begin() + 6, 1);
  result = thrust::reduce(h_data.begin(), h_data.begin() + 6, 1);
  result = thrust::reduce(thrust::host, h_data.begin(), h_data.begin() + 6, -1, thrust::maximum<int>());
  result = thrust::reduce(h_data.begin(), h_data.begin() + 6, -1, thrust::maximum<int>());
  result = thrust::reduce(thrust::cuda::par(thrust_allocator), d_data.begin(), d_data.begin() + 6, -1, thrust::maximum<int>());
}
