// RUN: dpct --format-range=none -out-root %T/vector_add %s --cuda-include-path="%cuda-path/include" -- -std=c++17  -x cuda --cuda-host-only
// RUN: dpct --format-range=none --enable-codepin -out-root %T/vector_add %s --cuda-include-path="%cuda-path/include" -- -std=c++17  -x cuda --cuda-host-only
// RUN: cat %S/dpct_output_ref/vectorAdd_build_in_type.dp.cpp >> %T/vector_add/vectorAdd_build_in_type.dp.cpp
// RUN: cat %S/dpct_output_codepin_cuda_ref/vectorAdd_build_in_type.cu >> %T/vector_add_codepin_cuda/vectorAdd_build_in_type.cu
// RUN: cat %S/dpct_output_codepin_sycl_ref/vectorAdd_build_in_type.dp.cpp >> %T/vector_add_codepin_sycl/vectorAdd_build_in_type.dp.cpp
// RUN: FileCheck --match-full-lines --input-file %T/vector_add_codepin_cuda/vectorAdd_build_in_type.cu %T/vector_add_codepin_cuda/vectorAdd_build_in_type.cu
// RUN: FileCheck --match-full-lines --input-file %T/vector_add/vectorAdd_build_in_type.dp.cpp %T/vector_add/vectorAdd_build_in_type.dp.cpp
// RUN: FileCheck --match-full-lines --input-file %T/vector_add_codepin_sycl/vectorAdd_build_in_type.dp.cpp %T/vector_add_codepin_sycl/vectorAdd_build_in_type.dp.cpp

//==============================================================
// Copyright 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

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

  cudaMalloc(&d_A, VECTOR_SIZE * sizeof(float));
  cudaMalloc(&d_B, VECTOR_SIZE * sizeof(float));
  cudaMalloc(&d_C, VECTOR_SIZE * sizeof(float));
  VectorAddKernel<<<1, VECTOR_SIZE>>>(d_A, d_B, d_C);
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
