// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0
// RUN: dpct --format-range=none -out-root %T/thrust-complex %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/thrust-complex/thrust-complex.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/thrust-complex/thrust-complex.dp.cpp -o %T/thrust-complex/thrust-complex.dp.o %}
// CHECK: #include <oneapi/dpl/execution>
// CHECK-NEXT: #include <oneapi/dpl/algorithm>
// CHECK-EMPTY:
// CHECK-NEXT: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <dpct/dpl_utils.hpp>
// CHECK-NEXT: #include <complex>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/complex.h>

template<typename T>
// CHECK: std::complex<T> foo(std::complex<T> cp) {
thrust::complex<T> foo(thrust::complex<T> cp) {
// CHECK:   std::complex<T> c = std::complex<T>(0.0);
  thrust::complex<T> c = thrust::complex<T>(0.0);
  return c;
}

void bar(thrust::complex<double> *p);

// CHECK: void kernel(std::complex<double> *p, std::complex<double> c, std::complex<double> *d,
// CHECK-NEXT:  std::complex<sycl::double2> *s) {
__global__ void kernel(thrust::complex<double> *p, thrust::complex<double> c, thrust::complex<double> *d) {
  __shared__ thrust::complex<struct double2> s[10];
}

// CHECK: void template_kernel(T *, std::complex<T> *s) {
template<class T>
__global__ void template_kernel(T *) {
  __shared__ thrust::complex<T> s[10];
}

template<typename T>
class C {
  T data;
public:
  C();
  C(const C &c);
};

template<template <typename> class TT, typename T>
class CC {
  TT<T> data;
};

template<typename T> C<T> F();

int main() {
// CHECK:   std::complex<float> cf = foo(std::complex<float>(1.0));
  thrust::complex<float> cf = foo(thrust::complex<float>(1.0));
// CHECK:   std::complex<double> cd = foo(std::complex<double>(1.0));
  thrust::complex<double> cd = foo(thrust::complex<double>(1.0));
// CHECK:   std::complex<float> log = std::log(cf);
  thrust::complex<float> log = thrust::log(cf);
// CHECK:   std::complex<double> exp = std::exp(cd);
  thrust::complex<double> exp = thrust::exp(cd);
// CHECK:   dpct::device_pointer<std::complex<double>> dc_ptr = dpct::malloc_device<std::complex<double>>(1);
  thrust::device_ptr<thrust::complex<double>> dc_ptr = thrust::device_malloc<thrust::complex<double>>(1);

// CHECK:   C<std::complex<double>> c1 = F<std::complex<double>>();
  C<thrust::complex<double>> c1 = F<thrust::complex<double>>();
// CHECK:   C<std::complex<double> *> c2 = F<std::complex<double> *>();
  C<thrust::complex<double> *> c2 = F<thrust::complex<double> *>();
// CHECK:   C<std::complex<double> &> c3 = F<std::complex<double> &>();
  C<thrust::complex<double> &> c3 = F<thrust::complex<double> &>();

// CHECK:   C<C<std::complex<double>>> c4;
  C<C<thrust::complex<double>>> c4;

//  TODO: Use of non-specialized template types
//  CC<thrust::complex, double> c4;

// Check that no warnings are issued when using complex operators
// CHECK:   cf = cf + 1.0f;
  cf = cf + 1.0f;
// CHECK:   cf = cf - 1.0f;
  cf = cf - 1.0f;
// CHECK:   cf = cf * 1.0f;
  cf = cf * 1.0f;
// CHECK:   cf = cf / 1.0f;
  cf = cf / 1.0f;
// CHECK:   bool b1 = (cf == 1.0f);
  bool b1 = (cf == 1.0f);
// CHECK:  std::complex<float> cf2;
// CHECK-NEXT:   bool b2 = (cf != cf2);
  thrust::complex<float> cf2;
  bool b2 = (cf != cf2);

// Check migration of template types when used in reinterpret_cast/static_cast
  std::complex<double> *cdp;
// CHECK:   std::complex<double> *tcdp = reinterpret_cast<std::complex<double> *>(cdp);
  thrust::complex<double> *tcdp = reinterpret_cast<thrust::complex<double> *>(cdp);
// CHECK:   std::complex<double> cd2 = static_cast<std::complex<double>>(*cdp);
  thrust::complex<double> cd2 = static_cast<thrust::complex<double>>(*cdp);
// CHECK:   bar(reinterpret_cast<std::complex<double> *>(cdp));
  bar(reinterpret_cast<thrust::complex<double> *>(cdp));
  // CHECK:   q_ct1.submit(
  // CHECK-NEXT:     [&](sycl::handler &cgh) {
  // CHECK-NEXT:       sycl::local_accessor<std::complex<sycl::double2>, 1> s_acc_ct1(sycl::range<1>(10), cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:       std::complex<double> static_cast_thrust_complex_double_cdp_ct1 = static_cast<std::complex<double>>(*cdp);
  // CHECK-NEXT:       std::complex<double> * thrust_raw_pointer_cast_dc_ptr_ct2 = dpct::get_raw_pointer(dc_ptr);
  // CHECK-EMPTY:
  // CHECK-NEXT:       cgh.parallel_for(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 256), sycl::range<3>(1, 1, 256)),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           kernel(reinterpret_cast<std::complex<double> *>(cdp), static_cast_thrust_complex_double_cdp_ct1, thrust_raw_pointer_cast_dc_ptr_ct2, s_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  kernel<<<1, 256>>>(reinterpret_cast<thrust::complex<double> *>(cdp), static_cast<thrust::complex<double>>(*cdp), thrust::raw_pointer_cast(dc_ptr));

  int *d_i;
// CHECK:   q_ct1.submit(
// CHECK-NEXT:     [&](sycl::handler &cgh) {
// CHECK-NEXT:       sycl::local_accessor<std::complex<int>, 1> s_acc_ct1(sycl::range<1>(10), cgh);
// CHECK-EMPTY:
// CHECK-NEXT:       cgh.parallel_for(
// CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 256), sycl::range<3>(1, 1, 256)),
// CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:           template_kernel<int>(d_i, s_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
// CHECK-NEXT:         });
// CHECK-NEXT:     });
  template_kernel<int><<<1, 256>>>(d_i);

  return 0;
}

#include <thrust/reduce.h>
//CHECK: int foo(){
//CHECK-NEXT:   double *p;
//CHECK-NEXT:   dpct::device_pointer<double> dp(p);
//CHECK-NEXT:   double sum = std::reduce(oneapi::dpl::execution::make_device_policy(dpct::get_in_order_queue()), dp, dp + 10);
//CHECK-NEXT:   printf("sum = %f\n", sum);
//CHECK-NEXT:   std::complex<float> c1(1.0);
//CHECK-NEXT:   std::complex<float> c2(2.0);
//CHECK-NEXT:   std::complex<float> c3 = c1 + c2;
//CHECK-NEXT:   printf("c1 + c2 = (%f, %f)\n", c3.real(), c3.imag());
//CHECK-NEXT: }
int foo(){
  double *p;
  thrust::device_ptr<double> dp(p);
  double sum = thrust::reduce(dp, dp + 10);
  printf("sum = %f\n", sum);
  thrust::complex<float> c1(1.0);
  thrust::complex<float> c2(2.0);
  thrust::complex<float> c3 = c1 + c2;
  printf("c1 + c2 = (%f, %f)\n", c3.real(), c3.imag());
}

