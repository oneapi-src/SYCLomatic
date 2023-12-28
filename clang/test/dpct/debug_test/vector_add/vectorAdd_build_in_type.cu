//==============================================================
// Copyright � 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include "dpct/codepin/codepin.hpp"
#include "generated_schema.hpp"
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
  dpct::experimental::gen_prolog_API_CP("vectorAdd:vecotr.cu:[29]:", 0, TYPE_SHCEMA_005, (long *)&d_A, TYPE_SHCEMA_006, (long *)&d_B,  TYPE_SHCEMA_007, (long *)&d_C);
  VectorAddKernel<<<1, VECTOR_SIZE>>>(d_A, d_B, d_C);
  dpct::experimental::gen_epilog_API_CP("vectorAdd:vecotr.cu:[29]:", 0, TYPE_SHCEMA_005, (long *)&d_A, TYPE_SHCEMA_006, (long *)&d_B, TYPE_SHCEMA_007, (long *)&d_C);

  float Result[VECTOR_SIZE] = {};

  // dpct::experimental::gen_prolog_API_CP("cudaMemcpy:vecotr.cu:[237]:", 0, TYPE_SHCEMA_004, (long *)&h_C, (size_t)size, TYPE_SHCEMA_007, (long *)&d_C, (size_t)size);
  status = cudaMemcpy(Result, d_C, VECTOR_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
  // dpct::experimental::gen_epilog_API_CP("cudaMemcpy:vecotr.cu:[237]:", 0, TYPE_SHCEMA_004, (long *)&h_C, (size_t)size, TYPE_SHCEMA_007, (long *)&d_C, (size_t)size);

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
