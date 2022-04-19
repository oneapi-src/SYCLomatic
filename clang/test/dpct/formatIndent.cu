// RUN: cat %s > %T/formatIndent.cu
// RUN: cd %T
// RUN: dpct -out-root %T/formatIndent formatIndent.cu --cuda-include-path="%cuda-path/include" -- -std=c++14  -x cuda --cuda-host-only
// RUN: FileCheck -strict-whitespace formatIndent.cu --match-full-lines --input-file %T/formatIndent/formatIndent.dp.cpp

#include <cuda_runtime.h>

//If do not add this foo function, the GuessIndentWidthRule will match the line 41,
//then the indent width will be 18.
void foo(){
    int a;
    int b;
}


//CHECK:void foo1(){
//CHECK-NEXT:                  //some comments
//CHECK-NEXT:    //some comments
//CHECK-NEXT:    sycl::range<3> griddim = sycl::range<3>(1, 1, 2);
//CHECK-NEXT:}
void foo1(){
                  //some comments
    //some comments
    dim3 griddim = 2;
}

//CHECK:void foo2(){
//CHECK-NEXT:                //some comments
//CHECK-NEXT:    sycl::range<3> griddim = sycl::range<3>(1, 1, 2);
//CHECK-NEXT:}
void foo2(){
                //some comments
    dim3 griddim = 2;
}

//CHECK:void foo3(){
//CHECK-NEXT:                  int test;
//CHECK-NEXT:    sycl::range<3> griddim = sycl::range<3>(1, 1, 2);
//CHECK-NEXT:}
void foo3(){
                  int test;
    dim3 griddim = 2;
}