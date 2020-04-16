// RUN: dpct -out-root %T %s --cuda-include-path="%cuda-path/include" --format-range=none -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/insert_extra_args.dp.cpp --match-full-lines %s

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
  //CHECK: dpct::get_default_queue().submit(
  //CHECK-NEXT:   [&](sycl::handler &cgh) {
  //CHECK-NEXT:     cgh.parallel_for(
  //CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 2), sycl::range<3>(1, 1, 2)),
  //CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  //CHECK-NEXT:           kernelFoo(1, 2, item_ct1);
  //CHECK-NEXT:         });
  //CHECK-NEXT:   });
  kernelFoo<<<1,2>>>(1,2);
  //CHECK: dpct::get_default_queue().submit(
  //CHECK-NEXT:   [&](sycl::handler &cgh) {
  //CHECK-NEXT:     cgh.parallel_for(
  //CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 2), sycl::range<3>(1, 1, 2)),
  //CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  //CHECK-NEXT:           kernelFoo1(1, item_ct1, 2);
  //CHECK-NEXT:         });
  //CHECK-NEXT:   });
  kernelFoo1<<<1,2>>>(1,2);
  //CHECK: dpct::get_default_queue().submit(
  //CHECK-NEXT:   [&](sycl::handler &cgh) {
  //CHECK-NEXT:     cgh.parallel_for(
  //CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 2), sycl::range<3>(1, 1, 2)),
  //CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  //CHECK-NEXT:           kernelFoo2(item_ct1, 1, 2);
  //CHECK-NEXT:         });
  //CHECK-NEXT:   });
  kernelFoo2<<<1,2>>>(1,2);
  return 0;
}

