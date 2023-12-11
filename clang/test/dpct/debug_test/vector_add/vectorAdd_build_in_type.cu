/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include "dpct/debug/debug_helper.hpp"
#include "generated_schema.hpp"


/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void vectorAdd(const float *A, const float *B, float *C,
                          int numElements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < numElements) {
    C[i] = A[i] + B[i] + 0.0f;
  }
}

/**
 * Host main routine
 */
int main(void) {
  // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;

  // Print the vector length to be used, and compute its size
  int numElements = 5;
  size_t size = numElements * sizeof(float);
  printf("[Vector addition of %d elements]\n", numElements);

  // Allocate the host input vector A
  float *h_A = (float *)malloc(size);

  // Allocate the host input vector B
  float *h_B = (float *)malloc(size);

  // Allocate the host output vector C
  float *h_C = (float *)malloc(size);
  
  // Verify that allocations succeeded
  if (h_A == NULL || h_B == NULL || h_C == NULL) {
    fprintf(stderr, "Failed to allocate host vectors!\n");
    exit(EXIT_FAILURE);
  }

  // Initialize the host input vectors
  for (int i = 0; i < numElements; ++i) {
    h_A[i] = rand() / (float)RAND_MAX;
    h_B[i] = rand() / (float)RAND_MAX;
  }

      // Allocate the device input vector A
  float *d_A = NULL;
  err = cudaMalloc((void **)&d_A, size);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Allocate the device input vector B
  float *d_B = NULL;
  err = cudaMalloc((void **)&d_B, size);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Allocate the device output vector C
  float *d_C = NULL;
  err = cudaMalloc((void **)&d_C, size);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy the host input vectors A and B in host memory to the device input
  // vectors in
  // device memory
  printf("Copy input data from the host memory to the CUDA device\n");
 // target API, device pointer to host, copy then wait host.
 // Kernel function: parameter (1st value of the point.) // 
 // (d_A, 1)
 dpct::experimental::gen_prolog_API_CP("cudaMemcpy:vecotr.cu:[185]:", 0, TYPE_SHCEMA_005, (long *)&d_A, (size_t)size, TYPE_SHCEMA_002, (long *)&h_A, (size_t)size);
 err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
 dpct::experimental::gen_epilog_API_CP("cudaMemcpy:vecotr.cu:[185]:", 0, TYPE_SHCEMA_005, (long *)&d_A, (size_t)size, TYPE_SHCEMA_002, (long *)&h_A, (size_t)size);


  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector A from host to device (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  dpct::experimental::gen_prolog_API_CP("cudaMemcpy:vecotr.cu:[196]:", 0, TYPE_SHCEMA_006, (long *)&d_B, (size_t)size, TYPE_SHCEMA_003, (long *)&h_B, (size_t)size);
  err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
  dpct::experimental::gen_epilog_API_CP("cudaMemcpy:vecotr.cu:[196]:", 0, TYPE_SHCEMA_006, (long *)&d_B, (size_t)size, TYPE_SHCEMA_003, (long *)&h_B, (size_t)size);

  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector B from host to device (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Launch the Vector Add CUDA Kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
         threadsPerBlock);
  dpct::experimental::gen_prolog_API_CP("vectorAdd:vecotr.cu:[221]:", 0, TYPE_SHCEMA_005, (long *)&d_A, dpct::experimental::get_size_of_schema(TYPE_SHCEMA_005), TYPE_SHCEMA_006, (long *)&d_B, dpct::experimental::get_size_of_schema(TYPE_SHCEMA_006), TYPE_SHCEMA_007, (long *)&d_C, dpct::experimental::get_size_of_schema(TYPE_SHCEMA_007), TYPE_SHCEMA_008, (long *)&numElements, dpct::experimental::get_size_of_schema(TYPE_SHCEMA_008));
  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
  dpct::experimental::gen_epilog_API_CP("vectorAdd:vecotr.cu:[221]:", 0, TYPE_SHCEMA_005, (long *)&d_A, dpct::experimental::get_size_of_schema(TYPE_SHCEMA_005), TYPE_SHCEMA_006, (long *)&d_B, dpct::experimental::get_size_of_schema(TYPE_SHCEMA_006), TYPE_SHCEMA_007, (long *)&d_C, dpct::experimental::get_size_of_schema(TYPE_SHCEMA_007), TYPE_SHCEMA_008, (long *)&numElements, dpct::experimental::get_size_of_schema(TYPE_SHCEMA_008));

  err = cudaGetLastError();

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy the device result vector in device memory to the host result vector
  // in host memory.
  printf("Copy output data from the CUDA device to the host memory\n");
 dpct::experimental::gen_prolog_API_CP("cudaMemcpy:vecotr.cu:[237]:", 0, TYPE_SHCEMA_004, (long *)&h_C, (size_t)size, TYPE_SHCEMA_007, (long *)&d_C, (size_t)size);
  err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
 dpct::experimental::gen_epilog_API_CP("cudaMemcpy:vecotr.cu:[237]:", 0, TYPE_SHCEMA_004, (long *)&h_C, (size_t)size, TYPE_SHCEMA_007, (long *)&d_C, (size_t)size);


  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector C from device to host (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Verify that the result vector is correct
  for (int i = 0; i < numElements; ++i) {
    if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
      fprintf(stderr, "Result verification failed at element %d!\n", i);
      exit(EXIT_FAILURE);
    }
  }

  printf("Test PASSED\n");

  // Free device global memory
  err = cudaFree(d_A);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector A (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_B);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector B (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_C);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector C (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Free host memory
  free(h_A);
  free(h_B);
  free(h_C);

  printf("Done\n");
  return 0;
}
