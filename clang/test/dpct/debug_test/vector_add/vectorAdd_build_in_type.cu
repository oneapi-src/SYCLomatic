// RUN: dpct --format-range=none --enable-codepin -out-root %T/debug_test/vector_add %s --cuda-include-path="%cuda-path/include" -- -std=c++17  -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/debug_test/vector_add/vectorAdd_build_in_type.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/debug_test/vector_add/vectorAdd_build_in_type.dp.cpp -o %T/debug_test/vector_add/vectorAdd_build_in_type.dp.o %}
//==============================================================
// Copyright 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

//CHECK: #include <dpct/codepin/codepin.hpp>
//CHECK: #include "generated_schema.hpp"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#define VECTOR_SIZE 256

__global__ void VectorAddKernel(float *A, float *B, float *C) {
  A[threadIdx.x] = threadIdx.x + 1.0f;
  B[threadIdx.x] = threadIdx.x + 1.0f;
  C[threadIdx.x] = A[threadIdx.x] + B[threadIdx.x];
}

int main() {
  float *d_A, *d_B, *d_C;
  cudaError_t status;

  //CHECK: dpct::experimental::get_ptr_size_map()[d_A] = VECTOR_SIZE * sizeof(float);
  cudaMalloc(&d_A, VECTOR_SIZE * sizeof(float));
  //CHECK: dpct::experimental::get_ptr_size_map()[d_B] = VECTOR_SIZE * sizeof(float);
  cudaMalloc(&d_B, VECTOR_SIZE * sizeof(float));
  //CHECK: dpct::experimental::get_ptr_size_map()[d_C] = VECTOR_SIZE * sizeof(float);
  cudaMalloc(&d_C, VECTOR_SIZE * sizeof(float));
  //CHECK: dpct::experimental::gen_prolog_API_CP("{{[._0-9a-zA-Z\/\(\)\:]+}}", &q_ct1, "d_A", d_A, "d_B", d_B, "d_C", d_C);
  VectorAddKernel<<<1, VECTOR_SIZE>>>(d_A, d_B, d_C);
  //CHECK: dpct::experimental::gen_epilog_API_CP("{{[._0-9a-zA-Z\/\(\)\:]+}}", &q_ct1, "d_A", d_A, "d_B", d_B, "d_C", d_C);
  float Result[VECTOR_SIZE] = {};
 
  status = cudaMemcpy(Result, d_C, VECTOR_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
  
  if (status != cudaSuccess) {
    printf("Could not copy result to host\n");
    exit(EXIT_FAILURE);
  }

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  for (int i = 0; i < VECTOR_SIZE; i++) {
    if (i % 16 == 0) {
      printf("\n");
    }
    printf("%3.0f ", Result[i]);
  }
  printf("\n");

  return 0;
}
