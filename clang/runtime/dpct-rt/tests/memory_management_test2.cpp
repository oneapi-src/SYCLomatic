//===--- dpct_device.hpp ------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 - 2019 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//
//
// Here's a example of bash script, which can compile and run this example
// as regular SYCL program using ComputeCpp SDK:
//
// #!/bin/bash
// NAME=memory_management_test1
// #GCC_ABI_HACK=-D_GLIBCXX_USE_CXX11_ABI=0
// compute++ -std=c++14 -O2 -mllvm -inline-threshold=1000 -sycl -intelspirmetadata -emit-llvm -isystem /include/ -I/include/ -I/opt/intel/opencl/include -o $NAME.cpp.sycl -c $NAME.cpp $GCC_ABI_HACK &&
// c++ -isystem /include -isystem /opt/intel/opencl/include -Wall -include $NAME.cpp.sycl -std=gnu++14 -o $NAME.cpp.o -c $NAME.cpp  $GCC_ABI_HACK &&
// c++ -Wall $NAME.cpp.o  -o $NAME -rdynamic /lib/libComputeCpp.so -lOpenCL -Wl,-rpath,/lib: $GCC_ABI_HACK &&
// ./$NAME
//
#include <CL/sycl.hpp>
#include "../include/dpct.hpp"

dpct::device_memory<volatile int, 0> g_a(dpct::range<0>(), 0);
dpct::device_memory<int, 1> d_a(36);


void test2(volatile int &a) {
  a = 3;
}

void test1(dpct::accessor<volatile int, dpct::device, 0> acc_a, dpct::accessor<int, dpct::device, 1> acc_b) {
  unsigned d_a = 1;
  acc_a = 0;
  acc_a = d_a;
  unsigned d_c = (unsigned)acc_a;
  unsigned *d_d = (unsigned *)acc_b;
  test2(acc_a);
}

int main() try {
  {
    dpct::get_default_queue().submit(
      [&](cl::sycl::handler &cgh) {
        auto g_acc = g_a.get_access(cgh);
        auto d_acc = d_a.get_access(cgh);
        cgh.parallel_for<dpct_kernel_name<class kernel_test>>(
          cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, 1), cl::sycl::range<3>(1, 1, 1)),
          [=] (cl::sycl::nd_item<3> item) {
            test1(dpct::accessor<volatile int, dpct::device, 0>(g_acc), dpct::accessor<int, dpct::device, 1>(d_acc));
          });
      });
  }

  return 0;
}
catch(cl::sycl::exception const &exc){}
