// UNSUPPORTED: -linux-
// RUN: dpct --format-range=none --usm-level=none -out-root %T/template-deduce_windows %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -fno-delayed-template-parsing
// RUN: FileCheck %s --match-full-lines --input-file %T/template-deduce_windows/template-deduce_windows.dp.cpp

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

    // CHECK:  sycl::accessor<TemplateClass<T2, T1>, 1, sycl::access_mode::read_write, sycl::access::target::local> a_acc_ct1(sycl::range<1>(10), cgh);
    // CHECK-EMPTY:
    // CHECK-NEXT:  cgh.parallel_for(
    // CHECK-NEXT:    sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:    [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:      template_kernel1<T2, T1>(a_acc_ct1.get_pointer());
    // CHECK-NEXT:    });
    template_kernel1<T2, T1><<<1,1>>>();

    // CHECK:  sycl::accessor<TemplateClass<T2, int>, 1, sycl::access_mode::read_write, sycl::access::target::local> a_acc_ct1(sycl::range<1>(10), cgh);
    // CHECK-EMPTY:
    // CHECK-NEXT:  cgh.parallel_for(
    // CHECK-NEXT:    sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:    [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:      template_kernel2<T2>(a_acc_ct1.get_pointer());
    // CHECK-NEXT:    });
    template_kernel2<T2><<<1,1>>>();

    // CHECK:  sycl::accessor<TemplateClass<T1, T1>, 1, sycl::access_mode::read_write, sycl::access::target::local> a_acc_ct1(sycl::range<1>(S), cgh);
    // CHECK-EMPTY:
    // CHECK-NEXT:  cgh.parallel_for(
    // CHECK-NEXT:    sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:    [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:      template_kernel3<T1, S>(a_acc_ct1.get_pointer());
    // CHECK-NEXT:    });
    template_kernel3<T1, S><<<1,1>>>();

    // CHECK:  sycl::accessor<TemplateClass<T2, float>, 1, sycl::access_mode::read_write, sycl::access::target::local> a_acc_ct1(sycl::range<1>(3), cgh);
    // CHECK-EMPTY:
    // CHECK-NEXT:  cgh.parallel_for(
    // CHECK-NEXT:    sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:    [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:      template_kernel4<T2>(a_acc_ct1.get_pointer());
    // CHECK-NEXT:    });
    template_kernel4<T2><<<1,1>>>();
}

int main() {

    // CHECK:  sycl::accessor<TemplateClass<sycl::float4, int>, 1, sycl::access_mode::read_write, sycl::access::target::local> a_acc_ct1(sycl::range<1>(10), cgh);
    // CHECK-EMPTY:
    // CHECK-NEXT:  cgh.parallel_for(
    // CHECK-NEXT:    sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:    [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:      template_kernel1<sycl::float4, int>(a_acc_ct1.get_pointer());
    // CHECK-NEXT:    });
    template_kernel1<float4, int><<<1,1>>>();

    // CHECK:  sycl::accessor<TemplateClass<sycl::float2, int>, 1, sycl::access_mode::read_write, sycl::access::target::local> a_acc_ct1(sycl::range<1>(10), cgh);
    // CHECK-EMPTY:
    // CHECK-NEXT:  cgh.parallel_for(
    // CHECK-NEXT:    sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:    [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:      template_kernel2<sycl::float2>(a_acc_ct1.get_pointer());
    // CHECK-NEXT:    });
    template_kernel2<float2><<<1,1>>>();

    // CHECK:  sycl::accessor<TemplateClass<sycl::float2, sycl::float2>, 1, sycl::access_mode::read_write, sycl::access::target::local> a_acc_ct1(sycl::range<1>(3), cgh);
    // CHECK-EMPTY:
    // CHECK-NEXT:  cgh.parallel_for(
    // CHECK-NEXT:    sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:    [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:      template_kernel3<sycl::float2, 3>(a_acc_ct1.get_pointer());
    // CHECK-NEXT:    });
    template_kernel3<float2, 3><<<1,1>>>();

    // CHECK:  sycl::accessor<TemplateClass<sycl::float2, float>, 1, sycl::access_mode::read_write, sycl::access::target::local> a_acc_ct1(sycl::range<1>(3), cgh);
    // CHECK-EMPTY:
    // CHECK-NEXT:  cgh.parallel_for(
    // CHECK-NEXT:    sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:    [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:      template_kernel4<sycl::float2>(a_acc_ct1.get_pointer());
    // CHECK-NEXT:    });
    template_kernel4<float2><<<1,1>>>();
}

