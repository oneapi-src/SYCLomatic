//===--- syclct_device.hpp ------------------------------*- C++ -*---===//
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
#include "../include/syclct_memory.hpp"

void test1() {

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
  // hostA[0..999] -> deviceA[0..999]
  // hostB[0..3999] -> deviceA[1000..4999]
  // deviceA[0..4999] -> hostC[0..4999]
  syclct::sycl_malloc((void **)&d_A, Num * sizeof(float));
  syclct::sycl_memcpy((void*) d_A, (void*) h_A, N1 * sizeof(float), syclct::memcpy_direction::host_to_device);
  syclct::sycl_memcpy((void*) (d_A + N1), (void*) h_B, (Num-N1) * sizeof(float), syclct::memcpy_direction::host_to_device);
  syclct::sycl_memcpy((void*) h_C, (void*) d_A, Num * sizeof(float), syclct::memcpy_direction::device_to_host);
  syclct::sycl_free((void*)d_A);

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

  printf("Test1 Passed\n");

  free(h_A);
  free(h_B);
  free(h_C);
}

void test1_1() {

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
  // hostA[0..999] -> deviceA[0..999]
  // hostB[0..3999] -> deviceA[1000..4999]
  // deviceA[0..4999] -> hostC[0..4999]
  syclct::sycl_malloc((void **)&d_A, Num * sizeof(float));
  syclct::sycl_memcpy((void*) d_A, (void*) h_A, N1 * sizeof(float), syclct::memcpy_direction::automatic);
  syclct::sycl_memcpy((void*) (d_A + N1), (void*) h_B, (Num-N1) * sizeof(float), syclct::memcpy_direction::automatic);
  syclct::sycl_memcpy((void*) h_C, (void*) d_A, Num * sizeof(float), syclct::memcpy_direction::automatic);
  syclct::sycl_free((void*)d_A);

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

  printf("Test1.1 Passed\n");

  free(h_A);
  free(h_B);
  free(h_C);
}

// TODO: corresponding API is to be removed from SYCLCT. We keep this code for
// now to be able file bug agains OpenCL CPU runtime.
/*
class vectorAdd2;
void test2() {

  int Num = 5000;
  int Offset = 32;
  float *h_A = (float*)malloc(Num*sizeof(float));
  float *h_B = (float*)malloc(Num*sizeof(float));
  float *h_C = (float*)malloc(Num*sizeof(float));

  for (int i = 0; i < Num; i++) {
    h_A[i] = 1.0f;
    h_B[i] = 2.0f;
  }

  float *d_A, *d_B, *d_C;
  // hostA -> deviceA
  // hostB -> deviceB
  // kernel: deviceC = deviceA + deviceB
  // deviceA -> hostC
  syclct::sycl_malloc((void **)&d_A, Num * sizeof(float));
  syclct::sycl_malloc((void **)&d_B, Num * sizeof(float));
  syclct::sycl_malloc((void **)&d_C, Num * sizeof(float));
  syclct::sycl_memcpy((void*) d_A, (void*) h_A, Num * sizeof(float), syclct::memcpy_direction::host_to_device);
  syclct::sycl_memcpy((void*) d_B, (void*) h_B, Num * sizeof(float), syclct::memcpy_direction::host_to_device);

  d_A += Offset;
  d_B += Offset;
  d_C += Offset;

  {
    syclct::get_device_manager().current_device().default_queue().submit(
      [=](cl::sycl::handler &cgh) {
      syclct::buffer_t buffer_A = syclct::get_buffer(d_A);
      syclct::buffer_t buffer_B = syclct::get_buffer(d_B);
      syclct::buffer_t buffer_C = syclct::get_buffer(d_C);

      auto d_A_acc = buffer_A.template get_access<cl::sycl::access::mode::read_write>(cgh);
      auto d_B_acc = buffer_B.template get_access<cl::sycl::access::mode::read_write>(cgh);
      auto d_C_acc = buffer_C.template get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.parallel_for<class vectorAdd2>(
          cl::sycl::range<1>(Num-Offset),
          [=](cl::sycl::id<1> id) {
            float *A = (float*)&d_A_acc[0];
            float *B = (float*)&d_B_acc[0];
            float *C = (float*)&d_C_acc[0];
            int i = id[0];
            C[i] = A[i] + B[i];
          });
      });
  }

  syclct::sycl_memcpy((void*) (h_C+Offset), (void*) d_C, (Num-Offset) * sizeof(float), syclct::memcpy_direction::device_to_host);

  syclct::sycl_free((void*)d_A);
  syclct::sycl_free((void*)d_B);
  syclct::sycl_free((void*)d_C);

  // verify
  for(int i = Offset; i < Num; i++){
      if (fabs(h_C[i] - h_A[i] - h_B[i]) > 1e-5) {
          fprintf(stderr,"Check: Elements are A = %f, B = %f, C = %f:\n", h_A[i],  h_B[i],  h_C[i]);
          fprintf(stderr,"Result verification failed at element %d:\n", i);
          exit(EXIT_FAILURE);
      }
  }

  printf("Test2 Passed\n");

  free(h_A);
  free(h_B);
  free(h_C);
}
*/

class vectorAdd3;
void test3() {

  int Num = 5000;
  int Offset = 100;
  float *h_A = (float*)malloc(Num*sizeof(float));
  float *h_B = (float*)malloc(Num*sizeof(float));
  float *h_C = (float*)malloc(Num*sizeof(float));

  for (int i = 0; i < Num; i++) {
    h_A[i] = 1.0f;
    h_B[i] = 2.0f;
  }

  float *d_A, *d_B, *d_C;
  // hostA -> deviceA
  // hostB -> deviceB
  // kernel: deviceC = deviceA + deviceB
  // deviceA -> hostC
  syclct::sycl_malloc((void **)&d_A, Num * sizeof(float));
  syclct::sycl_malloc((void **)&d_B, Num * sizeof(float));
  syclct::sycl_malloc((void **)&d_C, Num * sizeof(float));
  syclct::sycl_memcpy((void*) d_A, (void*) h_A, Num * sizeof(float), syclct::memcpy_direction::host_to_device);
  syclct::sycl_memcpy((void*) d_B, (void*) h_B, Num * sizeof(float), syclct::memcpy_direction::host_to_device);

  d_A += Offset;
  d_B += Offset;
  d_C += Offset;

  {
    syclct::get_device_manager().current_device().default_queue().submit(
      [=](cl::sycl::handler &cgh) {
      auto buffer_and_offset_A = syclct::get_buffer_and_offset(d_A);
      size_t offset_A = buffer_and_offset_A.second;
      auto buffer_and_offset_B = syclct::get_buffer_and_offset(d_B);
      size_t offset_B = buffer_and_offset_A.second;
      auto buffer_and_offset_C = syclct::get_buffer_and_offset(d_C);
      size_t offset_C = buffer_and_offset_A.second;

      auto d_A_acc = buffer_and_offset_A.first.template get_access<cl::sycl::access::mode::read_write>(cgh);
      auto d_B_acc = buffer_and_offset_B.first.template get_access<cl::sycl::access::mode::read_write>(cgh);
      auto d_C_acc = buffer_and_offset_C.first.template get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.parallel_for<class vectorAdd3>(
          cl::sycl::range<1>(Num-Offset),
          [=](cl::sycl::id<1> id) {
            float *A = (float*)(&d_A_acc[0]+offset_A);
            float *B = (float*)(&d_B_acc[0]+offset_B);
            float *C = (float*)(&d_C_acc[0]+offset_C);
            int i = id[0];
            C[i] = A[i] + B[i];
          });
      });
  }

  syclct::sycl_memcpy((void*) (h_C+Offset), (void*) d_C, (Num-Offset) * sizeof(float), syclct::memcpy_direction::device_to_host);

  syclct::sycl_free((void*)d_A);
  syclct::sycl_free((void*)d_B);
  syclct::sycl_free((void*)d_C);

  // verify
  for(int i = Offset; i < Num; i++){
      if (fabs(h_C[i] - h_A[i] - h_B[i]) > 1e-5) {
          fprintf(stderr,"Check: Elements are A = %f, B = %f, C = %f:\n", h_A[i],  h_B[i],  h_C[i]);
          fprintf(stderr,"Result verification failed at element %d:\n", i);
          exit(EXIT_FAILURE);
      }
  }

  printf("Test3 Passed\n");

  free(h_A);
  free(h_B);
  free(h_C);
}


void test4() {

  int Num = 10;
  int *h_A = (int*)malloc(Num*sizeof(int));

  for (int i = 0; i < Num; i++) {
    h_A[i] = 4;
  }

  int *d_A;

  syclct::sycl_malloc((void **)&d_A, Num * sizeof(int));
  // hostA -> deviceA
  syclct::sycl_memcpy((void*) d_A, (void*) h_A, Num * sizeof(int), syclct::memcpy_direction::host_to_device);

  // set d_A[0,..., 6] = 0
  syclct::sycl_memset((void*) d_A, 0, (Num - 3) * sizeof(int));

  // deviceA -> hostA
  syclct::sycl_memcpy((void*) h_A, (void*) d_A, Num * sizeof(int), syclct::memcpy_direction::device_to_host);

  syclct::sycl_free((void*)d_A);

  // check d_A[0,..., 6] = 0
  for (int i = 0; i < Num - 3; i++) {
    if (h_A[i] != 0) {
      fprintf(stderr, "Check: h_A[%d] is %d:\n", i, h_A[i]);
      fprintf(stderr, "Result verification failed at element [%d]!\n", i);
      exit(EXIT_FAILURE);
    }
  }

  // check d_A[7,..., 9] = 4
  for (int i = Num - 3; i < Num; i++) {
    if (h_A[i] != 4) {
      fprintf(stderr, "Check: h_A[%d] is %d:\n", i, h_A[i]);
      fprintf(stderr, "Result verification failed at element h_A[%d]!\n", i);
      exit(EXIT_FAILURE);
    }
  }

  printf("Test4 Passed\n");

  free(h_A);
}

const unsigned int Num = 5000;
const unsigned int N1 = 1000;
syclct::ConstMem d_A(Num * sizeof(float));
syclct::ConstMem d_B(Num * sizeof(float));

void test5() {

  float h_A[Num];
  float h_B[Num];
  float h_C[Num];
  float h_D[Num];

  for (int i = 0; i < Num; i++) {
    h_A[i] = 1.0f;
    h_B[i] = 2.0f;
  }

  for (int i = 0; i < Num; i++) {
    h_A[i] = 1.0f;
    h_B[i] = 2.0f;
  }
  // hostA[0..999] -> deviceA[0..999]
  // hostB[0..3999] -> deviceA[1000..4999]
  // deviceA[0..4999] -> deviceB[0..4999]
  // deviceA[0..4999] -> hostC[0..4999]
  // deviceB[0..4999] -> hostD[0..4999]

  syclct::sycl_memcpy_to_symbol((void *)d_A.get_ptr(), (void *)&h_A[0], N1 * sizeof(float), 0, syclct::memcpy_direction::host_to_device);
  syclct::sycl_memcpy_to_symbol(d_A.get_ptr(), (void*) h_B, (Num-N1) * sizeof(float), N1 * sizeof(float), syclct::memcpy_direction::automatic);
  syclct::sycl_memcpy((void *)h_C, (void *)d_A.get_ptr(), Num * sizeof(float),   syclct::memcpy_direction::device_to_host);

  syclct::sycl_memcpy_to_symbol((void *)d_B.get_ptr(), (void *)d_A.get_ptr(), N1 * sizeof(float), 0, syclct::memcpy_direction::device_to_device);
  syclct::sycl_memcpy_to_symbol((void *)d_B.get_ptr(), (void *)((size_t)d_A.get_ptr() + N1* sizeof(float)), (Num - N1) * sizeof(float), N1 * sizeof(float), syclct::memcpy_direction::automatic);
  syclct::sycl_memcpy((void *)h_D, (void *)d_B.get_ptr(), Num * sizeof(float),   syclct::memcpy_direction::device_to_host);

  // verify hostD
  for (int i = 0; i < N1; i++) {
    if (fabs(h_A[i] - h_D[i]) > 1e-5) {
      fprintf(stderr, "Check: Elements are A = %f, D = %f:\n", h_A[i], h_D[i]);
      fprintf(stderr, "Result verification failed at element %d:\n", i);
      exit(EXIT_FAILURE);
    }
  }

  for (int i = N1; i < Num; i++) {
    if (fabs(h_B[i] - h_D[i]) > 1e-5) {
      fprintf(stderr, "Check: Elements are B = %f, D = %f:\n",   h_B[i], h_D[i]);
      fprintf(stderr, "Result verification failed at element %d:\n", i);
      exit(EXIT_FAILURE);
    }
  }

  printf("Test5 Passed\n");
}

int main() {
  test1();
  test1_1();
// Test fails on the latest OpenCL CPU runtime and the latest ComputeCpp 1.0.1
// The problem is caused by sub-buffer allocation. Looks like we need to
// depricate this method.
//  test2();
  test3();
  test4();
  test5();

  return 0;
}
