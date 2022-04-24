// RUN: dpct -out-root %T/kernel_indent_length %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck -strict-whitespace --input-file %T/kernel_indent_length/kernel_indent_length.dp.cpp --match-full-lines %s

#include <cuda.h>
#include <cstdio>


__global__ void k() {}

//     CHECK:void foo1(){
//CHECK-NEXT:    dpct::get_default_queue().parallel_for(
//CHECK-NEXT:        sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
//CHECK-NEXT:        [=](sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:            k();
//CHECK-NEXT:        });
//CHECK-NEXT:}
void foo1(){
  k<<<1,1>>>();
}


//     CHECK:void foo2(){
//CHECK-NEXT:    dpct::get_default_queue().parallel_for(
//CHECK-NEXT:        sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
//CHECK-NEXT:        [=](sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:            k();
//CHECK-NEXT:        });
//CHECK-NEXT:}
void foo2(){
    k<<<1,1>>>();
}

//     CHECK:void foo3(){
//CHECK-NEXT:    dpct::get_default_queue().parallel_for(
//CHECK-NEXT:        sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
//CHECK-NEXT:        [=](sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:            k();
//CHECK-NEXT:        });
//CHECK-NEXT:}
void foo3(){
    k<<<1,1>>>();
}

