// UNSUPPORTED: -linux-
// RUN: c2s -out-root %T/kernel-call-complex_windows %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only -fno-delayed-template-parsing
// RUN: FileCheck --input-file %T/kernel-call-complex_windows/kernel-call-complex_windows.dp.cpp --match-full-lines %s

__global__ void k(int i) {
}

template<typename T>
struct S {
  static int bar() {
    return 23;
  }
};

struct S2 {
  template<typename T>
  static T bar() {
    return (T)23;
  }
};

template<typename T>
T bar(T i) {
  return i;
}

template<typename T>
T bar() {
  return (T)23;
}

template<typename T>
void foo() {
  // CHECK: c2s::device_ext &dev_ct1 = c2s::get_current_device();
  // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.default_queue();
  T i;
  // CHECK:   q_ct1.submit(
  // CHECK-NEXT:       [&](sycl::handler &cgh) {
  // CHECK-NEXT:         auto bar_i_ct0 = bar(i);
  // CHECK-EMPTY:
  // CHECK-NEXT:         cgh.parallel_for<c2s_kernel_name<class k_{{[a-z0-9]+}}>>(
  // CHECK-NEXT:             sycl::nd_range<3>(sycl::range<3>(1, 1, 16) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
  // CHECK-NEXT:             [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:               k(bar_i_ct0);
  // CHECK-NEXT:             });
  // CHECK-NEXT:       });
  k<<<16, 32>>>(bar(i));

  // CHECK:   q_ct1.submit(
  // CHECK-NEXT:       [&](sycl::handler &cgh) {
  // CHECK-NEXT:         auto bar_T_ct0 = bar<T>();
  // CHECK-EMPTY:
  // CHECK-NEXT:         cgh.parallel_for<c2s_kernel_name<class k_{{[a-z0-9]+}}>>(
  // CHECK-NEXT:             sycl::nd_range<3>(sycl::range<3>(1, 1, 16) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
  // CHECK-NEXT:             [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:               k(bar_T_ct0);
  // CHECK-NEXT:             });
  // CHECK-NEXT:       });
  k<<<16, 32>>>(bar<T>());

  // CHECK:   q_ct1.submit(
  // CHECK-NEXT:       [&](sycl::handler &cgh) {
  // CHECK-NEXT:         auto S_T_bar_ct0 = S<T>::bar();
  // CHECK-EMPTY:
  // CHECK-NEXT:         cgh.parallel_for<c2s_kernel_name<class k_{{[a-z0-9]+}}>>(
  // CHECK-NEXT:             sycl::nd_range<3>(sycl::range<3>(1, 1, 16) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
  // CHECK-NEXT:             [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:               k(S_T_bar_ct0);
  // CHECK-NEXT:             });
  // CHECK-NEXT:       });
  k<<<16, 32>>>(S<T>::bar());

  // CHECK:   q_ct1.submit(
  // CHECK-NEXT:       [&](sycl::handler &cgh) {
  // CHECK-NEXT:         auto S2_bar_T_ct0 = S2::bar<T>();
  // CHECK-EMPTY:
  // CHECK-NEXT:         cgh.parallel_for<c2s_kernel_name<class k_{{[a-z0-9]+}}>>(
  // CHECK-NEXT:             sycl::nd_range<3>(sycl::range<3>(1, 1, 16) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
  // CHECK-NEXT:             [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:               k(S2_bar_T_ct0);
  // CHECK-NEXT:             });
  // CHECK-NEXT:       });
  k<<<16, 32>>>(S2::bar<T>());
}

