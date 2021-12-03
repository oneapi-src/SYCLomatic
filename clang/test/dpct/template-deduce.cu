// UNSUPPORTED: -windows-
// RUN: dpct --format-range=none -out-root %T/template-deduce %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++11
// RUN: FileCheck %s --match-full-lines --input-file %T/template-deduce/template-deduce.dp.cpp

#include <complex>

template<class T1, class T2> class TemplateClass { using type = T1; };

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
  // CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
  // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.default_queue();
  // CHECK: q_ct1.submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     sycl::accessor<int, 0, sycl::access_mode::read_write, sycl::access::target::local> v1_acc_ct1(cgh);
  // CHECK-NEXT:     sycl::accessor<double, 0, sycl::access_mode::read_write, sycl::access::target::local> v2_acc_ct1(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 16) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         k(1, 2.3, (int *)v1_acc_ct1.get_pointer(), (double *)v2_acc_ct1.get_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  k<<<16, 32>>>(1, 2.3);
  // CHECK: q_ct1.submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     sycl::accessor<int, 0, sycl::access_mode::read_write, sycl::access::target::local> v1_acc_ct1(cgh);
  // CHECK-NEXT:     sycl::accessor<double, 0, sycl::access_mode::read_write, sycl::access::target::local> v2_acc_ct1(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 16) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         k<int>(1, 2.3, v1_acc_ct1.get_pointer(), (double *)v2_acc_ct1.get_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  k<int><<<16, 32>>>(1, 2.3);
  // CHECK: q_ct1.submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     sycl::accessor<int, 0, sycl::access_mode::read_write, sycl::access::target::local> v1_acc_ct1(cgh);
  // CHECK-NEXT:     sycl::accessor<float, 0, sycl::access_mode::read_write, sycl::access::target::local> v2_acc_ct1(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 16) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         k<int, float>(1, 2.3, v1_acc_ct1.get_pointer(), v2_acc_ct1.get_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  k<int, float><<<16, 32>>>(1, 2.3);

  // CHECK: q_ct1.submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     sycl::accessor<T3, 0, sycl::access_mode::read_write, sycl::access::target::local> v1_acc_ct1(cgh);
  // CHECK-NEXT:     sycl::accessor<double, 0, sycl::access_mode::read_write, sycl::access::target::local> v2_acc_ct1(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 16) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         k<T3>(1, 2.3, v1_acc_ct1.get_pointer(), (double *)v2_acc_ct1.get_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  k<T3><<<16, 32>>>(1, 2.3);
  // CHECK: q_ct1.submit(
  // CHECK-NEXT:   [&](sycl::handler &cgh) {
  // CHECK-NEXT:     sycl::accessor<T3, 0, sycl::access_mode::read_write, sycl::access::target::local> v1_acc_ct1(cgh);
  // CHECK-NEXT:     sycl::accessor<T4, 0, sycl::access_mode::read_write, sycl::access::target::local> v2_acc_ct1(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:     cgh.parallel_for(
  // CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 16) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
  // CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:         k<T3, T4>(1, 2.3, v1_acc_ct1.get_pointer(), v2_acc_ct1.get_pointer());
  // CHECK-NEXT:       });
  // CHECK-NEXT:   });
  k<T3, T4><<<16, 32>>>(1, 2.3);
}

template<class T>
__global__ void kernel_ptr(T *b) { __shared__ T a[10]; }
template<class T, int N>
__global__ void kernel_array(T b[][N]) { __shared__ T a[N]; }
template<class T1, class T2>
__global__ void kernel_class(TemplateClass<T1, T2> b) {
  __shared__ T1 a1[10];
  __shared__ T2 a2[10];
}
template<class T>
__global__ void kernel_ref(const T &b) { __shared__ T a[10]; }
template<class T> __global__ void kernel_dependent(std::complex<T> *d) {
  __shared__ std::complex<T> a[10];
}

template<class T1, class T2, size_t S> void implicit_host() {
    typedef typename TemplateClass<T1, T2>::type typedef_1;
    using using_1 = typename TemplateClass<T1, T2>::type;
    
    T1 *d_a;
    int *d_b;
    typedef_1 *d_c;
    using_1 *d_d;

    // CHECK:  sycl::accessor<T1, 1, sycl::access_mode::read_write, sycl::access::target::local> a_acc_ct1(sycl::range<1>(10), cgh);
    // CHECK-EMPTY:
    // CHECK-NEXT:  cgh.parallel_for(
    // CHECK-NEXT:    sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:    [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:      kernel_ptr(d_a, (T1 *)a_acc_ct1.get_pointer());
    // CHECK-NEXT:    });
    kernel_ptr<<<1,1>>>(d_a);
    // CHECK:  sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::access::target::local> a_acc_ct1(sycl::range<1>(10), cgh);
    // CHECK-EMPTY:
    // CHECK-NEXT:  cgh.parallel_for(
    // CHECK-NEXT:    sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:    [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:      kernel_ptr(d_b, (int *)a_acc_ct1.get_pointer());
    // CHECK-NEXT:    });
    kernel_ptr<<<1,1>>>(d_b);
    // CHECK:  sycl::accessor<typedef_1, 1, sycl::access_mode::read_write, sycl::access::target::local> a_acc_ct1(sycl::range<1>(10), cgh);
    // CHECK-EMPTY:
    // CHECK-NEXT:  cgh.parallel_for(
    // CHECK-NEXT:    sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:    [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:      kernel_ptr(d_c, (typedef_1 *)a_acc_ct1.get_pointer());
    // CHECK-NEXT:    });
    kernel_ptr<<<1,1>>>(d_c);
    // CHECK:  sycl::accessor<using_1, 1, sycl::access_mode::read_write, sycl::access::target::local> a_acc_ct1(sycl::range<1>(10), cgh);
    // CHECK-EMPTY:
    // CHECK-NEXT:  cgh.parallel_for(
    // CHECK-NEXT:    sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:    [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:      kernel_ptr(d_d, (using_1 *)a_acc_ct1.get_pointer());
    // CHECK-NEXT:    });
    kernel_ptr<<<1,1>>>(d_d);
    
    T1 r_a;
    int r_b;
    typedef_1 r_c;
    using_1 r_d;

    // CHECK:  sycl::accessor<T1, 1, sycl::access_mode::read_write, sycl::access::target::local> a_acc_ct1(sycl::range<1>(10), cgh);
    // CHECK-EMPTY:
    // CHECK-NEXT:  cgh.parallel_for(
    // CHECK-NEXT:    sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:    [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:      kernel_ref(r_a, (T1 *)a_acc_ct1.get_pointer());
    // CHECK-NEXT:    });
    kernel_ref<<<1,1>>>(r_a);
    // CHECK:  sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::access::target::local> a_acc_ct1(sycl::range<1>(10), cgh);
    // CHECK-EMPTY:
    // CHECK-NEXT:  cgh.parallel_for(
    // CHECK-NEXT:    sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:    [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:      kernel_ref(r_b, (int *)a_acc_ct1.get_pointer());
    // CHECK-NEXT:    });
    kernel_ref<<<1,1>>>(r_b);
    // CHECK:  sycl::accessor<typedef_1, 1, sycl::access_mode::read_write, sycl::access::target::local> a_acc_ct1(sycl::range<1>(10), cgh);
    // CHECK-EMPTY:
    // CHECK-NEXT:  cgh.parallel_for(
    // CHECK-NEXT:    sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:    [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:      kernel_ref(r_c, (typedef_1 *)a_acc_ct1.get_pointer());
    // CHECK-NEXT:    });
    kernel_ref<<<1,1>>>(r_c);
    // CHECK:  sycl::accessor<using_1, 1, sycl::access_mode::read_write, sycl::access::target::local> a_acc_ct1(sycl::range<1>(10), cgh);
    // CHECK-EMPTY:
    // CHECK-NEXT:  cgh.parallel_for(
    // CHECK-NEXT:    sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:    [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:      kernel_ref(r_d, (using_1 *)a_acc_ct1.get_pointer());
    // CHECK-NEXT:    });
    kernel_ref<<<1,1>>>(r_d);

    T1 a_a[10][20];
    int a_b[10][20];
    typedef_1 a_c[10][20];
    using_1 a_d[10][20];
    // CHECK:  sycl::accessor<T1, 1, sycl::access_mode::read_write, sycl::access::target::local> a_acc_ct1(sycl::range<1>(20), cgh);
    // CHECK-EMPTY:
    // CHECK-NEXT:  cgh.parallel_for(
    // CHECK-NEXT:    sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:    [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:      kernel_array(a_a, (T1 *)a_acc_ct1.get_pointer());
    // CHECK-NEXT:    });
    kernel_array<<<1,1>>>(a_a);
    // CHECK:  sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::access::target::local> a_acc_ct1(sycl::range<1>(20), cgh);
    // CHECK-EMPTY:
    // CHECK-NEXT:  cgh.parallel_for(
    // CHECK-NEXT:    sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:    [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:      kernel_array(a_b, (int *)a_acc_ct1.get_pointer());
    // CHECK-NEXT:    });
    kernel_array<<<1,1>>>(a_b);
    // CHECK:  sycl::accessor<typedef_1, 1, sycl::access_mode::read_write, sycl::access::target::local> a_acc_ct1(sycl::range<1>(20), cgh);
    // CHECK-EMPTY:
    // CHECK-NEXT:  cgh.parallel_for(
    // CHECK-NEXT:    sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:    [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:      kernel_array(a_c, (typedef_1 *)a_acc_ct1.get_pointer());
    // CHECK-NEXT:    });
    kernel_array<<<1,1>>>(a_c);
    // CHECK:  sycl::accessor<using_1, 1, sycl::access_mode::read_write, sycl::access::target::local> a_acc_ct1(sycl::range<1>(20), cgh);
    // CHECK-EMPTY:
    // CHECK-NEXT:  cgh.parallel_for(
    // CHECK-NEXT:    sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:    [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:      kernel_array(a_d, (using_1 *)a_acc_ct1.get_pointer());
    // CHECK-NEXT:    });
    kernel_array<<<1,1>>>(a_d);
    
    using typedef_class = TemplateClass<T1, T2>;
    typedef_class c_a;
    TemplateClass<T1, T2> c_b;
    TemplateClass<T1, int> c_c;
    TemplateClass<int, T2> c_d;
    TemplateClass<int, double> c_e;

    // CHECK:  sycl::accessor<T1, 1, sycl::access_mode::read_write, sycl::access::target::local> a1_acc_ct1(sycl::range<1>(10), cgh);
    // CHECK-NEXT:  sycl::accessor<T2, 1, sycl::access_mode::read_write, sycl::access::target::local> a2_acc_ct1(sycl::range<1>(10), cgh);
    // CHECK-EMPTY:
    // CHECK-NEXT:  cgh.parallel_for(
    // CHECK-NEXT:    sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:    [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:      kernel_class(c_a, (T1 *)a1_acc_ct1.get_pointer(), (T2 *)a2_acc_ct1.get_pointer());
    // CHECK-NEXT:    });
    kernel_class<<<1,1>>>(c_a);
    // CHECK:  sycl::accessor<T1, 1, sycl::access_mode::read_write, sycl::access::target::local> a1_acc_ct1(sycl::range<1>(10), cgh);
    // CHECK-NEXT:  sycl::accessor<T2, 1, sycl::access_mode::read_write, sycl::access::target::local> a2_acc_ct1(sycl::range<1>(10), cgh);
    // CHECK-EMPTY:
    // CHECK-NEXT:  cgh.parallel_for(
    // CHECK-NEXT:    sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:    [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:      kernel_class(c_b, (T1 *)a1_acc_ct1.get_pointer(), (T2 *)a2_acc_ct1.get_pointer());
    // CHECK-NEXT:    });
    kernel_class<<<1,1>>>(c_b);
    // CHECK:  sycl::accessor<T1, 1, sycl::access_mode::read_write, sycl::access::target::local> a1_acc_ct1(sycl::range<1>(10), cgh);
    // CHECK-NEXT:  sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::access::target::local> a2_acc_ct1(sycl::range<1>(10), cgh);
    // CHECK-EMPTY:
    // CHECK-NEXT:  cgh.parallel_for(
    // CHECK-NEXT:    sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:    [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:      kernel_class(c_c, (T1 *)a1_acc_ct1.get_pointer(), (int *)a2_acc_ct1.get_pointer());
    // CHECK-NEXT:    });
    kernel_class<<<1,1>>>(c_c);
    // CHECK:  sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::access::target::local> a1_acc_ct1(sycl::range<1>(10), cgh);
    // CHECK-NEXT:  sycl::accessor<T2, 1, sycl::access_mode::read_write, sycl::access::target::local> a2_acc_ct1(sycl::range<1>(10), cgh);
    // CHECK-EMPTY:
    // CHECK-NEXT:  cgh.parallel_for(
    // CHECK-NEXT:    sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:    [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:      kernel_class(c_d, (int *)a1_acc_ct1.get_pointer(), (T2 *)a2_acc_ct1.get_pointer());
    // CHECK-NEXT:    });
    kernel_class<<<1,1>>>(c_d);
    // CHECK:  sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::access::target::local> a1_acc_ct1(sycl::range<1>(10), cgh);
    // CHECK-NEXT:  sycl::accessor<double, 1, sycl::access_mode::read_write, sycl::access::target::local> a2_acc_ct1(sycl::range<1>(10), cgh);
    // CHECK-EMPTY:
    // CHECK-NEXT:  cgh.parallel_for(
    // CHECK-NEXT:    sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:    [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:      kernel_class(c_e, (int *)a1_acc_ct1.get_pointer(), (double *)a2_acc_ct1.get_pointer());
    // CHECK-NEXT:    });
    kernel_class<<<1,1>>>(c_e);

    using typedef_2 = std::complex<double>;
    std::complex<int> *h_cpx;
    typedef_2 *h_cpx_2;
    // CHECK:  sycl::accessor<std::complex<int>, 1, sycl::access_mode::read_write, sycl::access::target::local> a_acc_ct1(sycl::range<1>(10), cgh);
    // CHECK-EMPTY:
    // CHECK-NEXT:  cgh.parallel_for(
    // CHECK-NEXT:    sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:    [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:      kernel_dependent(h_cpx, (std::complex<int> *)a_acc_ct1.get_pointer());
    // CHECK-NEXT:    });
    kernel_dependent<<<1,1>>>(h_cpx);
    // CHECK:  sycl::accessor<std::complex<double>, 1, sycl::access_mode::read_write, sycl::access::target::local> a_acc_ct1(sycl::range<1>(10), cgh);
    // CHECK-EMPTY:
    // CHECK-NEXT:  cgh.parallel_for(
    // CHECK-NEXT:    sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
    // CHECK-NEXT:    [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:      kernel_dependent(h_cpx_2, (std::complex<double> *)a_acc_ct1.get_pointer());
    // CHECK-NEXT:    });
    kernel_dependent<<<1,1>>>(h_cpx_2);
}

#include<vector>
template<class T1, class T2>
class MyClass{};

// Make sure no assert when there are default template arg.
// std::vector has a default arg class Allocator:
// template<class T, class Allocator = std::allocator<T>> class vector;
// When deducing std::vector<MyClass<T, T>, std::allocator<MyClass<T, T>>>
// from std::vector<T>, DPCT should use the #arg of std::vector<T>.
template<class T>
__device__ __host__ void foo2(
  std::vector<MyClass<T, T>> t){}

template<class T>
__device__ __host__ void foo3(){
  std::vector<T> a;
  foo2(a);
}

__global__ void foo4(){
  foo3<MyClass<int, int>>();
}

// Make sure stop deduction when the template names are different
// MyClass2 inheritates MyClass, but AST does not show the mapping of
// template args between MyClass and MyClass2.
// DPCT should stop the deduction to prevent potential issues.
template<class T, T N>
class MyClass2 : public MyClass<T, float>{};

template<class T>
__device__ __host__ void foo5(MyClass<T, float> t){}

template<class T>
__device__ __host__ void foo6(){
  MyClass2<T, 5> a;
  foo5(a);
}

__global__ void foo7(){
  foo6<int>();
}

// Make sure no assert when there are alias template deduction
// using MakeMyClass3 = MyClass3<T, 5> will create an alias template
// since MakeMyClass3 takes only 1 arg and MyClass3 takes 2 args,
// DPCT needs to perform getAliasedType() before deduction.
// Calling foo9<<<1,1,0>>>(m1) will trigger the deduction without typeloc
// Calling foo8(input) will trigger the deduction with typeloc
template<class T1, size_t T2>
class MyClass3{};

template<class T>
using MakeMyClass3 = MyClass3<T, 5>;

template<typename scalar_t>
__device__ inline void foo8(MakeMyClass3<scalar_t> input) {}

template<typename scalar_t>
__global__ void foo9(MakeMyClass3<scalar_t> input){
  foo8(input);
}

//CHECK: int foo10(){
//CHECK-NEXT:   MakeMyClass3<int> m1;
//CHECK-NEXT:   dpct::get_default_queue().parallel_for(
//CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
//CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:         foo9(m1);
//CHECK-NEXT:       });
//CHECK-NEXT:   return 0;
//CHECK-NEXT: }
int foo10(){
  MakeMyClass3<int> m1;
  foo9<<<1,1,0>>>(m1);
  return 0;
}