// UNSUPPORTED: -windows-
// RUN: dpct --format-range=none --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/template-deduce.dp.cpp

template<class T1, class T2> class TemplateClass {};

// CHECK: template<class T, size_t S> void template_device(T *a) {
// CHECK-EMPTY
template<class T, size_t S> __device__ void template_device() {
    __shared__ T a[S];
}

// CHECK: template<class T1, class T2> void template_kernel1(TemplateClass<T1, T2> *a) {
// CHECK-NEXT: template_device<TemplateClass<T1, T2>, 10>(a);
template<class T1, class T2> __global__ void template_kernel1() {
    template_device<TemplateClass<T1, T2>, 10>();
}

// CHECK: template<class T1> void template_kernel2(TemplateClass<T1, int> *a) {
// CHECK-NEXT: template_device<TemplateClass<T1, int>, 10>(a);
template<class T1> __global__ void template_kernel2() {
    template_device<TemplateClass<T1, int>, 10>();
}

// CHECK: template<class T, size_t S> void template_kernel3(TemplateClass<T, T> *a) {
// CHECK-NEXT: template_device<TemplateClass<T, T>, S>(a);
template<class T, size_t S> __global__ void template_kernel3() {
    template_device<TemplateClass<T, T>, S>();
}

// CHECK: template<class T> void template_kernel4(TemplateClass<T, float> *a) {
// CHECK-NEXT: template_device<TemplateClass<T, float>, 3>(a);
template<class T> __global__ void template_kernel4() {
    template_device<TemplateClass<T, float>, 3>();
}

template<class T1, class T2, size_t S> void template_host() {

    // CHECK:  sycl::accessor<TemplateClass<T2, T1>, 1, sycl::access::mode::read_write, sycl::access::target::local> a_acc_ct1(sycl::range<1>(10), cgh);
    // CHECK-EMPTY:
    // CHECK-NEXT:  cgh.parallel_for(
    // CHECK-NEXT:    sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:    [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:      template_kernel1<T2, T1>(a_acc_ct1.get_pointer());
    // CHECK-NEXT:    });
    template_kernel1<T2, T1><<<1,1>>>();

    // CHECK:  sycl::accessor<TemplateClass<T2, int>, 1, sycl::access::mode::read_write, sycl::access::target::local> a_acc_ct1(sycl::range<1>(10), cgh);
    // CHECK-EMPTY:
    // CHECK-NEXT:  cgh.parallel_for(
    // CHECK-NEXT:    sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:    [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:      template_kernel2<T2>(a_acc_ct1.get_pointer());
    // CHECK-NEXT:    });
    template_kernel2<T2><<<1,1>>>();

    // CHECK:  sycl::accessor<TemplateClass<T1, T1>, 1, sycl::access::mode::read_write, sycl::access::target::local> a_acc_ct1(sycl::range<1>(S), cgh);
    // CHECK-EMPTY:
    // CHECK-NEXT:  cgh.parallel_for(
    // CHECK-NEXT:    sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:    [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:      template_kernel3<T1, S>(a_acc_ct1.get_pointer());
    // CHECK-NEXT:    });
    template_kernel3<T1, S><<<1,1>>>();

    // CHECK:  sycl::accessor<TemplateClass<T2, float>, 1, sycl::access::mode::read_write, sycl::access::target::local> a_acc_ct1(sycl::range<1>(3), cgh);
    // CHECK-EMPTY:
    // CHECK-NEXT:  cgh.parallel_for(
    // CHECK-NEXT:    sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:    [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:      template_kernel4<T2>(a_acc_ct1.get_pointer());
    // CHECK-NEXT:    });
    template_kernel4<T2><<<1,1>>>();
}

int main() {

    // CHECK:  sycl::accessor<TemplateClass<sycl::float4, int>, 1, sycl::access::mode::read_write, sycl::access::target::local> a_acc_ct1(sycl::range<1>(10), cgh);
    // CHECK-EMPTY:
    // CHECK-NEXT:  cgh.parallel_for(
    // CHECK-NEXT:    sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:    [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:      template_kernel1<sycl::float4, int>(a_acc_ct1.get_pointer());
    // CHECK-NEXT:    });
    template_kernel1<float4, int><<<1,1>>>();

    // CHECK:  sycl::accessor<TemplateClass<sycl::float2, int>, 1, sycl::access::mode::read_write, sycl::access::target::local> a_acc_ct1(sycl::range<1>(10), cgh);
    // CHECK-EMPTY:
    // CHECK-NEXT:  cgh.parallel_for(
    // CHECK-NEXT:    sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:    [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:      template_kernel2<sycl::float2>(a_acc_ct1.get_pointer());
    // CHECK-NEXT:    });
    template_kernel2<float2><<<1,1>>>();

    // CHECK:  sycl::accessor<TemplateClass<sycl::float2, sycl::float2>, 1, sycl::access::mode::read_write, sycl::access::target::local> a_acc_ct1(sycl::range<1>(3), cgh);
    // CHECK-EMPTY:
    // CHECK-NEXT:  cgh.parallel_for(
    // CHECK-NEXT:    sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:    [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:      template_kernel3<sycl::float2, 3>(a_acc_ct1.get_pointer());
    // CHECK-NEXT:    });
    template_kernel3<float2, 3><<<1,1>>>();

    // CHECK:  sycl::accessor<TemplateClass<sycl::float2, float>, 1, sycl::access::mode::read_write, sycl::access::target::local> a_acc_ct1(sycl::range<1>(3), cgh);
    // CHECK-EMPTY:
    // CHECK-NEXT:  cgh.parallel_for(
    // CHECK-NEXT:    sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:    [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:      template_kernel4<sycl::float2>(a_acc_ct1.get_pointer());
    // CHECK-NEXT:    });
    template_kernel4<float2><<<1,1>>>();
}

// CHECK: template<typename T1, typename T2>
// CHECK-NEXT: void k(T1 arg1, T2 arg2, T1 *v1, T2 *v2) {
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK-NEXT:   *v1 += arg1;
// CHECK-NEXT:   *v2 += arg2;
// CHECK-NEXT: }
template<typename T1, typename T2>
__global__ void k(T1 arg1, T2 arg2) {
__shared__ T1 v1;
__shared__ T2 v2;
  v1 += arg1;
  v2 += arg2;
}

template<typename T3, typename T4>
void foo() {
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     sycl::accessor<T1, 0, sycl::access::mode::read_write, sycl::access::target::local> v1_acc_ct1(cgh);
  // CHECK-NEXT:     sycl::accessor<T2, 0, sycl::access::mode::read_write, sycl::access::target::local> v2_acc_ct1(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 16) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         k(1, 2.3, v1_acc_ct1.get_pointer(), v2_acc_ct1.get_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  k<<<16, 32>>>(1, 2.3);
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     sycl::accessor<int, 0, sycl::access::mode::read_write, sycl::access::target::local> v1_acc_ct1(cgh);
  // CHECK-NEXT:     sycl::accessor<PlaceHolder/*Fix the type mannually*/, 0, sycl::access::mode::read_write, sycl::access::target::local> v2_acc_ct1(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 16) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         k<int>(1, 2.3, v1_acc_ct1.get_pointer(), v2_acc_ct1.get_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  k<int><<<16, 32>>>(1, 2.3);
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     sycl::accessor<int, 0, sycl::access::mode::read_write, sycl::access::target::local> v1_acc_ct1(cgh);
  // CHECK-NEXT:     sycl::accessor<float, 0, sycl::access::mode::read_write, sycl::access::target::local> v2_acc_ct1(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 16) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         k<int, float>(1, 2.3, v1_acc_ct1.get_pointer(), v2_acc_ct1.get_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  k<int, float><<<16, 32>>>(1, 2.3);

  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     sycl::accessor<T3, 0, sycl::access::mode::read_write, sycl::access::target::local> v1_acc_ct1(cgh);
  // CHECK-NEXT:     sycl::accessor<PlaceHolder/*Fix the type mannually*/, 0, sycl::access::mode::read_write, sycl::access::target::local> v2_acc_ct1(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 16) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         k<T3>(1, 2.3, v1_acc_ct1.get_pointer(), v2_acc_ct1.get_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  k<T3><<<16, 32>>>(1, 2.3);
  // CHECK: dpct::get_default_queue().submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     sycl::accessor<T3, 0, sycl::access::mode::read_write, sycl::access::target::local> v1_acc_ct1(cgh);
  // CHECK-NEXT:     sycl::accessor<T4, 0, sycl::access::mode::read_write, sycl::access::target::local> v2_acc_ct1(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 16) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         k<T3, T4>(1, 2.3, v1_acc_ct1.get_pointer(), v2_acc_ct1.get_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  k<T3, T4><<<16, 32>>>(1, 2.3);
}
