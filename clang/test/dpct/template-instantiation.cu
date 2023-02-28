// RUN: dpct --format-range=none -out-root %T/template-instantiation %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++14
// RUN: FileCheck %s --match-full-lines --input-file %T/template-instantiation/template-instantiation.dp.cpp

#include <vector>

template<class T>
T &host_instantiation(T &a) { return a; }

// CHECK: template const std::vector<sycl::mfloat2> &host_instantiation(std::vector<sycl::mfloat2> const &);
template const std::vector<float2> &host_instantiation(std::vector<float2> const &);

// CHECK: void kernel(int *, const sycl::nd_item<3> &item_ct1, T *a);
template<class T>
__global__ void kernel(int *);

// CHECK: template void kernel<sycl::mfloat2>(int *, const sycl::nd_item<3> &item_ct1,
// CHECK-NEXT: sycl::mfloat2 *a);
template __global__ void kernel<float2>(int *);

// CHECK: template void kernel<int>(int *, const sycl::nd_item<3> &item_ct1, int *a);
template __global__ void kernel<int>(int *);

template<class T>
// CHECK: void kernel1(T *, const sycl::nd_item<3> &item_ct1, T *a) {
__global__ void kernel1(T *) {
  __shared__ T a[10];
  int b = blockDim.x;
}

// CHECK: template void kernel1(sycl::mchar4 *, const sycl::nd_item<3> &item_ct1,
// CHECK-NEXT:  sycl::mchar4 *a);
template __global__ void kernel1(char4 *);

// CHECK: template void kernel1<int>(int *, const sycl::nd_item<3> &item_ct1, int *a);
template __global__ void kernel1<int>(int *);

// CHECK: void kernel2(T1 *, T2 *, const sycl::nd_item<3> &item_ct1, T1 *a1, T2 *a2) {
template<class T1, class T2>
__global__ void kernel2(T1 *, T2 *) {
  __shared__ T1 a1[10];
  __shared__ T2 a2[10];
  int b = blockDim.x;
}

// CHECK: template void kernel2(sycl::mchar4 *, int *, const sycl::nd_item<3> &item_ct1,
// CHECK-NEXT: sycl::mchar4 *a1, int *a2);
template __global__ void kernel2(char4 *, int *);

// CHECK: template void kernel2<int>(int *, sycl::mfloat2 *, const sycl::nd_item<3> &item_ct1,
// CHECK-NEXT: int *a1, sycl::mfloat2 *a2);
template __global__ void kernel2<int>(int *, float2 *);

// CHECK: template void kernel2<sycl::mfloat2, sycl::mchar4>(sycl::mfloat2 *, sycl::mchar4 *,
// CHECK-NEXT:   const sycl::nd_item<3> &item_ct1,
// CHECK-NEXT:   sycl::mfloat2 *a1, sycl::mchar4 *a2);
template __global__ void kernel2<float2, char4>(float2 *, char4 *);

template<unsigned S, class T>
// CHECK: void kernel3(T *, const sycl::nd_item<3> &item_ct1, T *a) {
__global__ void kernel3(T *) {
  __shared__ T a[S];
  int b = blockDim.x;
}
// CHECK: template void kernel3<20>(int *, const sycl::nd_item<3> &item_ct1, int *a);
template __global__ void kernel3<20>(int *);

int main() {
    int *d;
    float2 *d1;
    int4 *d2;

// CHECK:      q_ct1.submit(
// CHECK-NEXT:   [&](sycl::handler &cgh) {
// CHECK-NEXT:     sycl::local_accessor<sycl::mfloat2, 1> a_acc_ct1(sycl::range<1>(10), cgh);
// CHECK-EMPTY:
// CHECK-NEXT:     cgh.parallel_for(
// CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
// CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:         kernel<sycl::mfloat2>(d, item_ct1, a_acc_ct1.get_pointer());
// CHECK-NEXT:     });
// CHECK-NEXT: });
    kernel<float2><<<1,1>>>(d);

// CHECK:      q_ct1.submit(
// CHECK-NEXT:   [&](sycl::handler &cgh) {
// CHECK-NEXT:     sycl::local_accessor<sycl::mint4, 1> a_acc_ct1(sycl::range<1>(10), cgh);
// CHECK-EMPTY:
// CHECK-NEXT:     cgh.parallel_for(
// CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
// CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:         kernel<sycl::mint4>(d, item_ct1, a_acc_ct1.get_pointer());
// CHECK-NEXT:     });
// CHECK-NEXT: });
    kernel<int4><<<1,1>>>(d);

// CHECK:      q_ct1.submit(
// CHECK-NEXT:   [&](sycl::handler &cgh) {
// CHECK-NEXT:     sycl::local_accessor<int, 1> a_acc_ct1(sycl::range<1>(10), cgh);
// CHECK-EMPTY:
// CHECK-NEXT:     cgh.parallel_for(
// CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
// CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:         kernel1(d, item_ct1, (int *)a_acc_ct1.get_pointer());
// CHECK-NEXT:     });
// CHECK-NEXT: });
    kernel1<<<1,1>>>(d);

// CHECK:      q_ct1.submit(
// CHECK-NEXT:   [&](sycl::handler &cgh) {
// CHECK-NEXT:     sycl::local_accessor<int, 1> a1_acc_ct1(sycl::range<1>(10), cgh);
// CHECK-NEXT:     sycl::local_accessor<sycl::mfloat2, 1> a2_acc_ct1(sycl::range<1>(10), cgh);
// CHECK-EMPTY:
// CHECK-NEXT:     cgh.parallel_for(
// CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
// CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:         kernel2<int>(d, d1, item_ct1, a1_acc_ct1.get_pointer(), (sycl::mfloat2 *)a2_acc_ct1.get_pointer());
// CHECK-NEXT:     });
// CHECK-NEXT: });
    kernel2<int><<<1,1>>>(d, d1);

// CHECK:      q_ct1.submit(
// CHECK-NEXT:   [&](sycl::handler &cgh) {
// CHECK-NEXT:     sycl::local_accessor<int, 1> a_acc_ct1(sycl::range<1>(20), cgh);
// CHECK-EMPTY:
// CHECK-NEXT:     cgh.parallel_for(
// CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
// CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:         kernel3<20>(d, item_ct1, (int *)a_acc_ct1.get_pointer());
// CHECK-NEXT:     });
// CHECK-NEXT: });
    kernel3<20><<<1,1>>>(d);
}

// CHECK: void kernel(int *, const sycl::nd_item<3> &item_ct1, T *a) {
template<class T>
__global__ void kernel(int *) {
  __shared__ T a[10];
  int b = blockDim.x;
}

