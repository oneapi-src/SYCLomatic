// UNSUPPORTED: -windows-
// RUN: dpct --format-range=none -out-root %T/module_kernel %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -ptx
// RUN: FileCheck %s --match-full-lines --input-file %T/module_kernel/module_kernel.dp.cpp

//CHECK: dpct::image_wrapper_base_p tex;
CUtexref tex;

//CHECK: extern "C" void foo(float* k, float* y, sycl::nd_item<3> item_ct1,
//CHECK-NEXT:     uint8_t *dpct_local, unsigned int *const_data);

//CHECK:extern "C" {
//CHECK-NEXT:void foo_wrapper(sycl::queue &queue, const sycl::nd_range<3> &nr, unsigned int localMemSize, void **kernelParams, void **extra);
//CHECK-NEXT:}
extern "C" __global__ void foo(float* k, float* y);


__constant__ unsigned int const_data[3] = {1, 2, 3};

//CHECK: void foo(float* k, float* y, sycl::nd_item<3> item_ct1, uint8_t *dpct_local,
//CHECK-NEXT:     unsigned int *const_data){
//CHECK-NEXT: auto s = (int *)dpct_local;
//CHECK-NEXT: unsigned int cd = const_data[2];
//CHECK-NEXT: int a = item_ct1.get_local_id(2);
//CHECK-NEXT: }

// CHECK:      extern "C" {
// CHECK-NEXT:   void foo_wrapper(sycl::queue &queue, const sycl::nd_range<3> &nr, unsigned int localMemSize, void **kernelParams, void **extra) {
// CHECK-NEXT:     using foo_type = decltype(foo);
// CHECK-NEXT:     using k_type = dpct::get_nth_parameter_t<0, foo_type>;
// CHECK-NEXT:     using y_type = dpct::get_nth_parameter_t<1, foo_type>;

// CHECK:          k_type *k_ptr;
// CHECK-NEXT:     y_type *y_ptr;
// CHECK-NEXT:     if (kernelParams) {
// CHECK-NEXT:       k_ptr = static_cast<k_type *>(kernelParams[0]);
// CHECK-NEXT:       y_ptr = static_cast<y_type *>(kernelParams[1]);
// CHECK-NEXT:     } else {
// CHECK-NEXT:       struct Args {
// CHECK-NEXT:         k_type k;
// CHECK-NEXT:         y_type y;
// CHECK-NEXT:       };
// CHECK-NEXT:       auto args_ptr = static_cast<Args *>(dpct::get_args_ptr(extra));
// CHECK-NEXT:       k_ptr = &args_ptr->k;
// CHECK-NEXT:       y_ptr = &args_ptr->y;
// CHECK-NEXT:     }
// CHECK-NEXT:     k_type& k = *k_ptr;
// CHECK-NEXT:     y_type& y = *y_ptr;

// CHECK:          queue.submit(
// CHECK-NEXT:       [&](sycl::handler &cgh) {
// CHECK-NEXT:         const_data.init(queue);
// CHECK:              auto const_data_ptr_ct1 = const_data.get_ptr();
// CHECK:              sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(sycl::range<1>(localMemSize), cgh);
// CHECK:              cgh.parallel_for(
// CHECK-NEXT:           nr,
// CHECK-NEXT:           [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:             foo(k, y, item_ct1, dpct_local_acc_ct1.get_pointer(), const_data_ptr_ct1);
// CHECK-NEXT:           });
// CHECK-NEXT:       });
// CHECK-NEXT:   }
// CHECK-NEXT: }
__global__ void foo(float* k, float* y){
    extern __shared__ int s[];
    unsigned int cd = const_data[2];
    int a = threadIdx.x;
}


//CHECK: void goo(){
//CHECK-NEXT:     float *a, *b;
//CHECK-NEXT:     dpct::get_default_queue().submit(
//CHECK-NEXT:       [&](sycl::handler &cgh) {
//CHECK-NEXT:         const_data.init();
//CHECK:         auto const_data_ptr_ct1 = const_data.get_ptr();
//CHECK:         sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(sycl::range<1>(0), cgh);
//CHECK:         cgh.parallel_for(
//CHECK-NEXT:           sycl::nd_range<3>(sycl::range<3>(1, 1, 2), sycl::range<3>(1, 1, 2)), 
//CHECK-NEXT:           [=](sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:             foo(a, b, item_ct1, dpct_local_acc_ct1.get_pointer(), const_data_ptr_ct1);
//CHECK-NEXT:           });
//CHECK-NEXT:       });
//CHECK-NEXT: }
void goo(){
    float *a, *b;
    foo<<<1,2>>>(a, b);
}
