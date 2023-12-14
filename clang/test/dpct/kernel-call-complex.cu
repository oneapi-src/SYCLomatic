// UNSUPPORTED: system-windows
// RUN: dpct -out-root %T/kernel-call-complex %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/kernel-call-complex/kernel-call-complex.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/kernel-call-complex/kernel-call-complex.dp.cpp -o %T/kernel-call-complex/kernel-call-complex.dp.o %}

__global__ void k(int i) {
}

__global__ void k2(int *p, int *p2) {
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
  // CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
  // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  T i;
  int *pointers[2];

  // CHECK:   q_ct1.submit(
  // CHECK-NEXT:       [&](sycl::handler &cgh) {
  // CHECK-NEXT:         int bar_i_ct0 = bar(i);
  // CHECK-EMPTY:
  // CHECK-NEXT:         cgh.parallel_for<dpct_kernel_name<class k_{{[a-z0-9]+}}>>(
  // CHECK-NEXT:             sycl::nd_range<3>(sycl::range<3>(1, 1, 16) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
  // CHECK-NEXT:             [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:               k(bar_i_ct0);
  // CHECK-NEXT:             });
  // CHECK-NEXT:       });
  k<<<16, 32>>>(bar(i));

  // CHECK:   q_ct1.submit(
  // CHECK-NEXT:       [&](sycl::handler &cgh) {
  // CHECK-NEXT:         int bar_T_ct0 = bar<T>();
  // CHECK-EMPTY:
  // CHECK-NEXT:         cgh.parallel_for<dpct_kernel_name<class k_{{[a-z0-9]+}}>>(
  // CHECK-NEXT:             sycl::nd_range<3>(sycl::range<3>(1, 1, 16) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
  // CHECK-NEXT:             [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:               k(bar_T_ct0);
  // CHECK-NEXT:             });
  // CHECK-NEXT:       });
  k<<<16, 32>>>(bar<T>());

  // CHECK:   q_ct1.submit(
  // CHECK-NEXT:       [&](sycl::handler &cgh) {
  // CHECK-NEXT:         int S_T_bar_ct0 = S<T>::bar();
  // CHECK-EMPTY:
  // CHECK-NEXT:         cgh.parallel_for<dpct_kernel_name<class k_{{[a-z0-9]+}}>>(
  // CHECK-NEXT:             sycl::nd_range<3>(sycl::range<3>(1, 1, 16) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
  // CHECK-NEXT:             [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:               k(S_T_bar_ct0);
  // CHECK-NEXT:             });
  // CHECK-NEXT:       });
  k<<<16, 32>>>(S<T>::bar());

  // CHECK:   q_ct1.submit(
  // CHECK-NEXT:       [&](sycl::handler &cgh) {
  // CHECK-NEXT:         int S2_bar_T_ct0 = S2::bar<T>();
  // CHECK-EMPTY:
  // CHECK-NEXT:         cgh.parallel_for<dpct_kernel_name<class k_{{[a-z0-9]+}}>>(
  // CHECK-NEXT:             sycl::nd_range<3>(sycl::range<3>(1, 1, 16) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
  // CHECK-NEXT:             [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:               k(S2_bar_T_ct0);
  // CHECK-NEXT:             });
  // CHECK-NEXT:       });
  k<<<16, 32>>>(S2::bar<T>());

  // CHECK: q_ct1.submit(
  // CHECK-NEXT:     [&](sycl::handler &cgh) {
  // CHECK-NEXT:       int *pointers_ct0 = pointers[0];
  // CHECK-NEXT:       int *pointers_ct1 = pointers[1];
  // CHECK-EMPTY:
  // CHECK-NEXT:         cgh.parallel_for<dpct_kernel_name<class k2_{{[a-z0-9]+}}>>(
  // CHECK-NEXT:             sycl::nd_range<3>(sycl::range<3>(1, 1, 16) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
  // CHECK-NEXT:             [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:               k2(pointers_ct0, pointers_ct1);
  // CHECK-NEXT:             });
  // CHECK-NEXT:     });
  k2<<<16, 32>>>(pointers[0], pointers[1]);
}

