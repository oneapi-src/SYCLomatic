// UNSUPPORTED: -linux-
// RUN: dpct --format-range=none -out-root %T/module_kernel_win %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -ptx
// RUN: FileCheck %s --match-full-lines --input-file %T/module_kernel_win/module_kernel_win.dp.cpp

//CHECK: dpct::image_wrapper_base_p tex;
CUtexref tex;

//CHECK: extern "C" void foo(float* k, float* y, sycl::nd_item<3> item_ct1, uint8_t *dpct_local);

//CHECK: extern "C" {
//CHECK-NEXT: __declspec(dllexport) void foo_wrapper(sycl::queue &queue, const sycl::nd_range<3> &nr, unsigned int localMemSize, void **kernelParams, void **extra);
//CHECK-NEXT: }
extern "C" __global__ void foo(float* k, float* y);


//CHECK: void foo(float* k, float* y, sycl::nd_item<3> item_ct1, uint8_t *dpct_local){
//CHECK-NEXT:     auto s = (int *)dpct_local;
//CHECK-NEXT:     int a = item_ct1.get_local_id(2);
//CHECK-NEXT: }

//CHECK: extern "C" {
//CHECK-NEXT: __declspec(dllexport) void foo_wrapper(sycl::queue &queue, const sycl::nd_range<3> &nr, unsigned int localMemSize, void **kernelParams, void **extra){
//CHECK-NEXT: float * k;
//CHECK-NEXT: float * y;
//CHECK-NEXT: if(kernelParams){
//CHECK-NEXT: k = (float *)kernelParams[0];
//CHECK-NEXT: y = (float *)kernelParams[1];
//CHECK-NEXT: } else {
//CHECK-NEXT: k = (float *)(extra + sizeof(void*));
//CHECK-NEXT: y = (float *)(extra + sizeof(void*) + sizeof(float *));
//CHECK-NEXT: }
//CHECK-NEXT: queue.submit(
//CHECK-NEXT:   [&](sycl::handler &cgh) {
//CHECK-NEXT:     sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(sycl::range<1>(localMemSize), cgh);
//CHECK:     cgh.parallel_for(
//CHECK-NEXT: nr,
//CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:         foo(k, y, item_ct1, dpct_local_acc_ct1.get_pointer());
//CHECK-NEXT:       });
//CHECK-NEXT:   });
//CHECK-NEXT: }
//CHECK-NEXT: }
__global__ void foo(float* k, float* y){
    extern __shared__ int s[];
    int a = threadIdx.x;
}


//CHECK: void goo(){
//CHECK-NEXT:     float *a, *b;
//CHECK-NEXT:     dpct::get_default_queue().submit(
//CHECK-NEXT:       [&](sycl::handler &cgh) {
//CHECK-NEXT:         sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(sycl::range<1>(0), cgh);

//CHECK:         cgh.parallel_for(
//CHECK-NEXT:           sycl::nd_range<3>(sycl::range<3>(1, 1, 2), sycl::range<3>(1, 1, 2)), 
//CHECK-NEXT:           [=](sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:             foo(a, b, item_ct1, dpct_local_acc_ct1.get_pointer());
//CHECK-NEXT:           });
//CHECK-NEXT:       });
//CHECK-NEXT: }
void goo(){
    float *a, *b;
    foo<<<1,2>>>(a, b);
}
