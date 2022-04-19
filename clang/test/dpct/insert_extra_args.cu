// RUN: dpct -out-root %T/insert_extra_args %s --cuda-include-path="%cuda-path/include" --format-range=none -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/insert_extra_args/insert_extra_args.dp.cpp --match-full-lines %s

//CHECK: void deviceFoo(int i, int j, sycl::nd_item<3> item_ct1){
//CHECK-NEXT: int a = item_ct1.get_group(2);
//CHECK-NEXT: }
__device__ void deviceFoo(int i, int j){
  int a = blockIdx.x;
}

//CHECK: void deviceFoo1(int i, sycl::nd_item<3> item_ct1, int j = 0){
//CHECK-NEXT:   int a = item_ct1.get_group(2);
//CHECK-NEXT: }
__device__ void deviceFoo1(int i, int j = 0){
  int a = blockIdx.x;
}

//CHECK: void deviceFoo2(sycl::nd_item<3> item_ct1, int i = 0, int j = 0){
//CHECK-NEXT:   int a = item_ct1.get_group(2);
//CHECK-NEXT: }
__device__ void deviceFoo2(int i = 0, int j = 0){
  int a = blockIdx.x;
}

//CHECK: void callDeviceFoo(sycl::nd_item<3> item_ct1){
//CHECK-NEXT:   deviceFoo(1, 2, item_ct1);
//CHECK-NEXT:   deviceFoo1(1, item_ct1, 2);
//CHECK-NEXT:   deviceFoo2(item_ct1, 1, 2);
//CHECK-NEXT: }
__global__ void callDeviceFoo(){
  deviceFoo(1, 2);
  deviceFoo1(1, 2);
  deviceFoo2(1, 2);
}

//CHECK: void kernelFoo(int i, int j, sycl::nd_item<3> item_ct1){
//CHECK-NEXT: int a = item_ct1.get_group(2);
//CHECK-NEXT: }
__global__ void kernelFoo(int i, int j){
  int a = blockIdx.x;
}

//CHECK: void kernelFoo1(int i, sycl::nd_item<3> item_ct1, int j = 0){
//CHECK-NEXT:   int a = item_ct1.get_group(2);
//CHECK-NEXT: }
__global__ void kernelFoo1(int i, int j = 0){
  int a = blockIdx.x;
}

//CHECK: void kernelFoo2(sycl::nd_item<3> item_ct1, int i = 0, int j = 0){
//CHECK-NEXT:   int a = item_ct1.get_group(2);
//CHECK-NEXT: }
__global__ void kernelFoo2(int i = 0, int j = 0){
  int a = blockIdx.x;
}

int main(){
  //CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
  //CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.default_queue();
  //CHECK: q_ct1.parallel_for(
  //CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 2), sycl::range<3>(1, 1, 2)),
  //CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  //CHECK-NEXT:           kernelFoo(1, 2, item_ct1);
  //CHECK-NEXT:         });
  kernelFoo<<<1,2>>>(1,2);
  //CHECK: q_ct1.parallel_for(
  //CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 2), sycl::range<3>(1, 1, 2)),
  //CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  //CHECK-NEXT:           kernelFoo1(1, item_ct1, 2);
  //CHECK-NEXT:         });
  kernelFoo1<<<1,2>>>(1,2);
  //CHECK: q_ct1.parallel_for(
  //CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 2), sycl::range<3>(1, 1, 2)),
  //CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  //CHECK-NEXT:           kernelFoo2(item_ct1, 1, 2);
  //CHECK-NEXT:         });
  kernelFoo2<<<1,2>>>(1,2);
  return 0;
}


typedef float Real_t;
#define VOLUDER(a0,a1,a2,a3,a4,a5,b0,b1,b2,b3,b4,b5,dvdc)	(a0,a1,a2,a3,a4,a5,b0,b1,b2,b3,b4,b5,dvdc)

//CHECK: float foo(float a, float b, int c, sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:  int i = item_ct1.get_group(2);
//CHECK-NEXT:  return 1.0f;
//CHECK-NEXT:}
__device__ float foo(float a, float b, int c) {
  int i = blockIdx.x;
  return 1.0f;
}

__global__ void bar() {
    float ind0, ind1, ind2, ind3, ind4, ind5, dvdzn;
    float xn, yn;

   //CHECK: VOLUDER(foo(xn,ind0,8, item_ct1),foo(xn,ind1,8, item_ct1),foo(xn,ind2,8, item_ct1),
   //CHECK-NEXT:         foo(xn,ind3,8, item_ct1),foo(xn,ind4,8, item_ct1),foo(xn,ind5,8, item_ct1),
   //CHECK-NEXT:         foo(yn,ind0,8, item_ct1),foo(yn,ind1,8, item_ct1),foo(yn,ind2,8, item_ct1),
   //CHECK-NEXT:         foo(yn,ind3,8, item_ct1),foo(yn,ind4,8, item_ct1),foo(yn,ind5,8, item_ct1),
   //CHECK-NEXT:       dvdzn);
    VOLUDER(foo(xn,ind0,8),foo(xn,ind1,8),foo(xn,ind2,8),
            foo(xn,ind3,8),foo(xn,ind4,8),foo(xn,ind5,8),
            foo(yn,ind0,8),foo(yn,ind1,8),foo(yn,ind2,8),
            foo(yn,ind3,8),foo(yn,ind4,8),foo(yn,ind5,8),
          dvdzn);
}

int foo_bar() {
    bar<<<1, 2>>>();
    return 0;
}


