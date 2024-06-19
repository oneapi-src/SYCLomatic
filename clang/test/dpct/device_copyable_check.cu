// RUN: dpct -out-root %T/device_copyable_check %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/device_copyable_check/device_copyable_check.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/device_copyable_check/device_copyable_check.dp.cpp -o %T/device_copyable_check/device_copyable_check.dp.o %}

class CudaImage {
public:
  float *d_data;
  float f_data;
  CudaImage() {};
  ~CudaImage() {};
};
__global__ void ScaleDown(float *x, float y) { }

template <typename T> struct S {
  T *t;
  S() {}
  ~S() {}
};
template <typename T> __global__ void g(const T *t) {}

template <typename T> void f(const S<T> &s) {
  // CHECK:  dpct::get_in_order_queue().submit(
  // CHECK-NEXT:    [&](sycl::handler &cgh) {
  // CHECK-NEXT:      auto s_t_ct0 = s.t;
  // CHECK-EMPTY:
  // CHECK-NEXT:      cgh.parallel_for(
  // CHECK-NEXT:          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:          [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:            g(s_t_ct0);
  // CHECK-NEXT:          });
  // CHECK-NEXT:    });
  g<<<1, 1>>>(s.t);
}

int main()
{
  CudaImage res;
  // CHECK: dpct::get_in_order_queue().submit(
  // CHECK-NEXT:     [&](sycl::handler &cgh) {
  // CHECK-NEXT:       auto res_d_data_ct0 = res.d_data;
  // CHECK-NEXT:       auto res_f_data_ct1 = res.f_data;
  // CHECK-EMPTY:
  // CHECK-NEXT:       cgh.parallel_for(
  // CHECK-NEXT:           sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:           [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:             ScaleDown(res_d_data_ct0, res_f_data_ct1);
  // CHECK-NEXT:           });
  // CHECK-NEXT:     });
  ScaleDown<<<1, 1>>>(res.d_data, res.f_data);
  f(S<float>());
  return 0;
}
