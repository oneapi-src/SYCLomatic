//===--- cu2sycl_device.hpp ------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
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
#include "../include/cu2sycl_memory.hpp"

int main() {

  int Num = 5000;
  int N1 = 1000;
  float *h_A = (float*)malloc(Num*sizeof(float));
  float *h_B = (float*)malloc(Num*sizeof(float));
  float *h_C = (float*)malloc(Num*sizeof(float));

  for (int i = 0; i < Num; i++) {
    h_A[i] = 1.0f;
    h_B[i] = 2.0f;
  }

  float *d_A;
  cu2sycl::sycl_malloc<float>((void **)&d_A, Num);
  cu2sycl::sycl_memcpy<float>((void*) h_A, (void*) d_A, N1, cu2sycl::memcpy_direction::to_device);
  cu2sycl::sycl_memcpy<float>((void*) h_B, (void*) (d_A + N1), Num-N1, cu2sycl::memcpy_direction::to_device);
  cu2sycl::sycl_memcpy<float>((void*) d_A, (void*) h_C, Num, cu2sycl::memcpy_direction::to_host);
  cu2sycl::sycl_free<float>((void*)d_A);

  // verify
  for(int i = 0; i < N1; i++){
      if (fabs(h_A[i] - h_C[i]) > 1e-5) {
          fprintf(stderr,"Check: Elements are A = %f, B = %f, C = %f:\n", h_A[i],  h_B[i],  h_C[i]);
          fprintf(stderr,"Result verification failed at element %d:\n", i);
          exit(EXIT_FAILURE);
      }
  }

  for(int i = N1; i < Num; i++){
      if (fabs(h_B[i] - h_C[i]) > 1e-5) {
          fprintf(stderr,"Check: Elements are A = %f, B = %f, C = %f:\n", h_A[i],  h_B[i],  h_C[i]);
          fprintf(stderr,"Result verification failed at element %d:\n", i);
          exit(EXIT_FAILURE);
      }
  }

  printf("Test Passed\n");

  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}
