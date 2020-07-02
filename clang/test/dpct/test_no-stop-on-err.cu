// RUN: dpct  --format-range=none -usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++14
#include <cuda.h>
#include <stdio.h>
#define VECTOR_SIZE 256

__global__ void VectorAddKernel(float* A, float* B, float* C); 

int main()
{
    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, VECTOR_SIZE*sizeof(float));
    cudaMalloc(&d_B, VECTOR_SIZE*sizeof(float));
    cudaMalloc(&d_C, VECTOR_SIZE*sizeof(float));

    VectorAddKernel<<<1, VECTOR_SIZE>>>(d_A, d_B, d_C);

    float Result[VECTOR_SIZE] = { };
    cudaMemcpy(Result, d_C, VECTOR_SIZE*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    for (int i = 0; i < VECTOR_SIZE; i++) {
        if (i % 16 == 0) {
            printf("\n");
        }
        printf("%f ", Result[i]);
    }

    return 0;
}
__global__ void VectorAddKernel(float* A, float* B, float* C)
{
    A[threadIdx.x] = threadIdx.x + 1.0f;
    B[threadIdx.x] = threadIdx.x + 1.0f;
    C[threadIdx.x] = A[threadIdx.x] + B[threadIdx.x];

    A[blockDim.x] = threadIdx.x + 1.0f;
    B[gridDim.x] = threadIdx.x + 1.0f;
    C[gridDim.x] = A[threadIdx.x] + B[threadIdx.x];

    __shared__  svar[10];

    A[threadIdx.x] = threadIdx.x + 1.0f;
    B[threadIdx.x] = threadIdx.x + 1.0f;
    C[threadIdx.x] = A[threadIdx.x] + B[threadIdx.x];
}

