// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -out-root %T/kernel_lambda_arg %s --usm-level=restricted --cuda-include-path="%cuda-path/include" --sycl-named-lambda
// RUN: FileCheck %s --match-full-lines --input-file %T/kernel_lambda_arg/kernel_lambda_arg.dp.cpp


//CHECK:template <typename T>
//CHECK-NEXT:void my_kernel1(const T func) {
//CHECK-NEXT:  func(10);
//CHECK-NEXT:}
template <typename T>
__global__ void my_kernel1(const T func) {
  func(10);
}

//CHECK:void run_foo1() {
//CHECK-NEXT:  dpct::get_default_queue().parallel_for<dpct_kernel_name<class my_kernel1_{{[0-9a-z]+}}, class lambda_{{[0-9a-z]+}}>>(
//CHECK-NEXT:    sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
//CHECK-NEXT:    [=](sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:      my_kernel1([=] (int idx) { idx++; });
//CHECK-NEXT:    });
//CHECK-NEXT:}
void run_foo1() {
  my_kernel1<<<1, 1>>>([=] __device__(int idx) { idx++; });
}