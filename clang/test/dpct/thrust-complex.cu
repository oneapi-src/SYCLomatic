// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0
// RUN: dpct --format-range=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/thrust-complex.dp.cpp --match-full-lines %s
// CHECK: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <dpct/dpstd_utils.hpp>
// CHECK-NEXT: #include <dpstd/execution>
// CHECK-NEXT: #include <dpstd/algorithm>
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

// CHECK: void kernel(std::complex<double> *p, std::complex<double> c,
// CHECK-NEXT:  std::complex<sycl::double2> *s) {
__global__ void kernel(thrust::complex<double> *p, thrust::complex<double> c) {
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
// CHECK:   dpct::device_ptr<std::complex<double>> dc_ptr = dpct::device_malloc<std::complex<double>>(1);
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
// CHECK:   cf = cf + 1.0;
  cf = cf + 1.0;
// CHECK:   cf = cf - 1.0;
  cf = cf - 1.0;
// CHECK:   cf = cf * 1.0;
  cf = cf * 1.0;
// CHECK:   cf = cf / 1.0;
  cf = cf / 1.0;
// CHECK:   bool b1 = (cf == 1.0);
  bool b1 = (cf == 1.0);
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
// CHECK-NEXT:       sycl::accessor<std::complex<sycl::double2>, 1, sycl::access::mode::read_write, sycl::access::target::local> s_acc_ct1(sycl::range<1>(10), cgh);
// CHECK-EMPTY:
// CHECK-NEXT:       cgh.parallel_for(
// CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 256), sycl::range<3>(1, 1, 256)),
// CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:           kernel(reinterpret_cast<std::complex<double> *>(cdp), static_cast<std::complex<double>>(*cdp), s_acc_ct1.get_pointer());
// CHECK-NEXT:         });
// CHECK-NEXT:     });
  kernel<<<1, 256>>>(reinterpret_cast<thrust::complex<double> *>(cdp), static_cast<thrust::complex<double>>(*cdp));

  int *d_i;
// CHECK:   q_ct1.submit(
// CHECK-NEXT:     [&](sycl::handler &cgh) {
// CHECK-NEXT:       sycl::accessor<std::complex<int>, 1, sycl::access::mode::read_write, sycl::access::target::local> s_acc_ct1(sycl::range<1>(10), cgh);
// CHECK-EMPTY:
// CHECK-NEXT:       cgh.parallel_for(
// CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 256), sycl::range<3>(1, 1, 256)),
// CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:           template_kernel<int>(d_i, s_acc_ct1.get_pointer());
// CHECK-NEXT:         });
// CHECK-NEXT:     });
  template_kernel<int><<<1, 256>>>(d_i);

  return 0;
}

#include <thrust/reduce.h>
//CHECK: int foo(){
//CHECK-NEXT:   double *p;
//CHECK-NEXT:   dpct::device_ptr<double> dp(p);
//CHECK-NEXT:   double sum = std::reduce(dpstd::execution::make_device_policy(dpct::get_default_queue()), dp, dp + 10);
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
