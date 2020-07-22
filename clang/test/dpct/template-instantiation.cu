// RUN: dpct --format-range=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++11
// RUN: FileCheck %s --match-full-lines --input-file %T/template-instantiation.dp.cpp

// CHECK: void kernel(int *, sycl::nd_item<3> item_ct1, T *a) {
template<class T>
__global__ void kernel(int *) {
  __shared__ T a[10];
  int b = blockDim.x;
}

// CHECK: template void kernel<sycl::float2>(int *, sycl::nd_item<3> item_ct1, sycl::float2 *a);
template __global__ void kernel<float2>(int *);

// CHECK: template void kernel<int>(int *, sycl::nd_item<3> item_ct1, int *a);
template __global__ void kernel<int>(int *);

template<class T>
// CHECK: void kernel1(T *, sycl::nd_item<3> item_ct1, T *a) {
__global__ void kernel1(T *) {
  __shared__ T a[10];
  int b = blockDim.x;
}

// CHECK: template void kernel1(sycl::char4 *, sycl::nd_item<3> item_ct1, sycl::char4 *a);
template __global__ void kernel1(char4 *);

// CHECK: template void kernel1<int>(int *, sycl::nd_item<3> item_ct1, int *a);
template __global__ void kernel1<int>(int *);

// CHECK: void kernel2(T1 *, T2 *, sycl::nd_item<3> item_ct1, T1 *a1, T2 *a2) {
template<class T1, class T2>
__global__ void kernel2(T1 *, T2 *) {
  __shared__ T1 a1[10];
  __shared__ T2 a2[10];
  int b = blockDim.x;
}

// CHECK: template void kernel2(sycl::char4 *, int *, sycl::nd_item<3> item_ct1,
// CHECK-NEXT: sycl::char4 *a1, int *a2);
template __global__ void kernel2(char4 *, int *);

// CHECK: template void kernel2<int>(int *, sycl::float2 *, sycl::nd_item<3> item_ct1, int *a1, 
// CHECK-NEXT: sycl::float2 *a2);
template __global__ void kernel2<int>(int *, float2 *);

// CHECK: template void kernel2<sycl::float2, sycl::char4>(sycl::float2 *, sycl::char4 *, 
// CHECK-NEXT:   sycl::nd_item<3> item_ct1, sycl::float2 *a1,
// CHECK-NEXT:   sycl::char4 *a2);
template __global__ void kernel2<float2, char4>(float2 *, char4 *);

template<unsigned S, class T>
// CHECK: void kernel3(T *, sycl::nd_item<3> item_ct1, T *a) {
__global__ void kernel3(T *) {
  __shared__ T a[S];
  int b = blockDim.x;
}
// CHECK: template void kernel3<20>(int *, sycl::nd_item<3> item_ct1, int *a);
template __global__ void kernel3<20>(int *);

int main() {
    int *d;
    float2 *d1;

// CHECK:      q_ct1.submit(
// CHECK-NEXT:   [&](sycl::handler &cgh) {
// CHECK-NEXT:     sycl::accessor<sycl::float2, 1, sycl::access::mode::read_write, sycl::access::target::local> a_acc_ct1(sycl::range<1>(10), cgh);
// CHECK-EMPTY:     
// CHECK-NEXT:     cgh.parallel_for(
// CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
// CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:         kernel<sycl::float2>(d, item_ct1, a_acc_ct1.get_pointer());
// CHECK-NEXT:     });
// CHECK-NEXT: });  
    kernel<float2><<<1,1>>>(d);
    
// CHECK:      q_ct1.submit(
// CHECK-NEXT:   [&](sycl::handler &cgh) {
// CHECK-NEXT:     sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::local> a_acc_ct1(sycl::range<1>(10), cgh);
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
// CHECK-NEXT:     sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::local> a1_acc_ct1(sycl::range<1>(10), cgh);
// CHECK-NEXT:     sycl::accessor<sycl::float2, 1, sycl::access::mode::read_write, sycl::access::target::local> a2_acc_ct1(sycl::range<1>(10), cgh);
// CHECK-EMPTY:     
// CHECK-NEXT:     cgh.parallel_for(
// CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
// CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:         kernel2<int>(d, d1, item_ct1, a1_acc_ct1.get_pointer(), (sycl::float2 *)a2_acc_ct1.get_pointer());
// CHECK-NEXT:     });
// CHECK-NEXT: });  
    kernel2<int><<<1,1>>>(d, d1);
    
// CHECK:      q_ct1.submit(
// CHECK-NEXT:   [&](sycl::handler &cgh) {
// CHECK-NEXT:     sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::local> a_acc_ct1(sycl::range<1>(20), cgh);
// CHECK-EMPTY:     
// CHECK-NEXT:     cgh.parallel_for(
// CHECK-NEXT:       sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
// CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:         kernel3<20>(d, item_ct1, (int *)a_acc_ct1.get_pointer());
// CHECK-NEXT:     });
// CHECK-NEXT: });  
    kernel3<20><<<1,1>>>(d);
}