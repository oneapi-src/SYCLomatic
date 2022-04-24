// RUN: dpct --format-range=none --no-cl-namespace-inline --usm-level=none -out-root %T/kernel-call-inner-virtual-pointer %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only -fno-delayed-template-parsing  -std=c++14

// RUN: FileCheck --input-file %T/kernel-call-inner-virtual-pointer/kernel-call-inner-virtual-pointer.dp.cpp --match-full-lines %s

#include "cuda_runtime.h"
#include <stdio.h>

struct AAA {
    int * a;
};


struct BBB {
  int i;
  struct CCC {
    int j;
    struct DDD {
      float* k;
    };
  };
};

struct EEE {
  int i;
  EEE *eee;
};

__global__ void k1(AAA a){}

__global__ void k2(AAA* a){
}

__global__ void k3(int **a){
}

__global__ void k4(BBB b){}

__global__ void k5(EEE e){}

int main() {
  int **a1;
  int **a2;
  cudaMalloc(a2, sizeof(int*));


  AAA a;
  AAA* b1;
  AAA* b2;
  cudaMalloc(&b2, sizeof(AAA));

  //CHECK:/*
  //CHECK-NEXT:DPCT1069:{{[0-9]+}}: The argument 'a' of the kernel function contains virtual pointer(s), which cannot be dereferenced. Try to migrate the code with "usm-level=restricted".
  //CHECK-NEXT:*/
  //CHECK-NEXT:q_ct1.parallel_for<dpct_kernel_name<class k1_{{[0-9a-z]+}}>>(
  //CHECK-NEXT:      cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, 1), cl::sycl::range<3>(1, 1, 1)),
  //CHECK-NEXT:      [=](cl::sycl::nd_item<3> item_ct1) {
  //CHECK-NEXT:        k1(a);
  //CHECK-NEXT:      });
  k1<<<1,1>>>(a);

  //CHECK:/*
  //CHECK-NEXT:DPCT1069:{{[0-9]+}}: The argument 'b1' of the kernel function contains virtual pointer(s), which cannot be dereferenced. Try to migrate the code with "usm-level=restricted".
  //CHECK-NEXT:*/
  //CHECK-NEXT:  q_ct1.submit(
  //CHECK-NEXT:    [&](cl::sycl::handler &cgh) {
  //CHECK-NEXT:      dpct::access_wrapper<AAA *> b1_acc_ct0(b1, cgh);
  //CHECK-EMPTY:
  //CHECK-NEXT:      cgh.parallel_for<dpct_kernel_name<class k2_{{[0-9a-z]+}}>>(
  //CHECK-NEXT:        cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, 1), cl::sycl::range<3>(1, 1, 1)),
  //CHECK-NEXT:        [=](cl::sycl::nd_item<3> item_ct1) {
  //CHECK-NEXT:          k2(b1_acc_ct0.get_raw_pointer());
  //CHECK-NEXT:        });
  //CHECK-NEXT:    });
  k2<<<1,1>>>(b1);

  //CHECK:/*
  //CHECK-NEXT:DPCT1069:{{[0-9]+}}: The argument 'b2' of the kernel function contains virtual pointer(s), which cannot be dereferenced. Try to migrate the code with "usm-level=restricted".
  //CHECK-NEXT:*/
  //CHECK-NEXT:  q_ct1.submit(
  //CHECK-NEXT:    [&](cl::sycl::handler &cgh) {
  //CHECK-NEXT:      auto b2_acc_ct0 = dpct::get_access(b2, cgh);
  //CHECK-EMPTY:
  //CHECK-NEXT:      cgh.parallel_for<dpct_kernel_name<class k2_{{[0-9a-z]+}}>>(
  //CHECK-NEXT:        cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, 1), cl::sycl::range<3>(1, 1, 1)),
  //CHECK-NEXT:        [=](cl::sycl::nd_item<3> item_ct1) {
  //CHECK-NEXT:          k2((AAA *)(&b2_acc_ct0[0]));
  //CHECK-NEXT:        });
  //CHECK-NEXT:    });
  k2<<<1,1>>>(b2);

  //CHECK:/*
  //CHECK-NEXT:DPCT1069:{{[0-9]+}}: The argument 'a1' of the kernel function contains virtual pointer(s), which cannot be dereferenced. Try to migrate the code with "usm-level=restricted".
  //CHECK-NEXT:*/
  //CHECK-NEXT:  q_ct1.submit(
  //CHECK-NEXT:    [&](cl::sycl::handler &cgh) {
  //CHECK-NEXT:      dpct::access_wrapper<int **> a1_acc_ct0(a1, cgh);
  //CHECK-EMPTY:
  //CHECK-NEXT:      cgh.parallel_for<dpct_kernel_name<class k3_{{[0-9a-z]+}}>>(
  //CHECK-NEXT:        cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, 1), cl::sycl::range<3>(1, 1, 1)),
  //CHECK-NEXT:        [=](cl::sycl::nd_item<3> item_ct1) {
  //CHECK-NEXT:          k3(a1_acc_ct0.get_raw_pointer());
  //CHECK-NEXT:        });
  //CHECK-NEXT:    });
  k3<<<1,1>>>(a1);

  //CHECK:/*
  //CHECK-NEXT:DPCT1069:{{[0-9]+}}: The argument 'a2' of the kernel function contains virtual pointer(s), which cannot be dereferenced. Try to migrate the code with "usm-level=restricted".
  //CHECK-NEXT:*/
  //CHECK-NEXT:  q_ct1.submit(
  //CHECK-NEXT:    [&](cl::sycl::handler &cgh) {
  //CHECK-NEXT:      dpct::access_wrapper<int **> a2_acc_ct0(a2, cgh);
  //CHECK-EMPTY:
  //CHECK-NEXT:      cgh.parallel_for<dpct_kernel_name<class k3_{{[0-9a-z]+}}>>(
  //CHECK-NEXT:        cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, 1), cl::sycl::range<3>(1, 1, 1)),
  //CHECK-NEXT:        [=](cl::sycl::nd_item<3> item_ct1) {
  //CHECK-NEXT:          k3(a2_acc_ct0.get_raw_pointer());
  //CHECK-NEXT:        });
  //CHECK-NEXT:    });
  k3<<<1,1>>>(a2);

  BBB b;
  //CHECK:/*
  //CHECK-NEXT:DPCT1069:{{[0-9]+}}: The argument 'b' of the kernel function contains virtual pointer(s), which cannot be dereferenced. Try to migrate the code with "usm-level=restricted".
  //CHECK-NEXT:*/
  //CHECK-NEXT:q_ct1.parallel_for<dpct_kernel_name<class k4_{{[0-9a-z]+}}>>(
  //CHECK-NEXT:      cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, 1), cl::sycl::range<3>(1, 1, 1)),
  //CHECK-NEXT:      [=](cl::sycl::nd_item<3> item_ct1) {
  //CHECK-NEXT:        k4(b);
  //CHECK-NEXT:      });
  k4<<<1,1>>>(b);

  EEE e;
  //CHECK:/*
  //CHECK-NEXT:DPCT1069:{{[0-9]+}}: The argument 'e' of the kernel function contains virtual pointer(s), which cannot be dereferenced. Try to migrate the code with "usm-level=restricted".
  //CHECK-NEXT:*/
  //CHECK-NEXT:q_ct1.parallel_for<dpct_kernel_name<class k5_{{[0-9a-z]+}}>>(
  //CHECK-NEXT:      cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, 1), cl::sycl::range<3>(1, 1, 1)),
  //CHECK-NEXT:      [=](cl::sycl::nd_item<3> item_ct1) {
  //CHECK-NEXT:        k5(e);
  //CHECK-NEXT:      });
  k5<<<1,1>>>(e);
}