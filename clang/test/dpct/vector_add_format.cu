// RUN: cat %s > %T/vector_add_format.cu
// RUN: cd %T
// RUN: dpct -out-root %T vector_add_format.cu --cuda-include-path="%cuda-path/include" -- -std=c++14  -x cuda --cuda-host-only
// RUN: FileCheck -strict-whitespace vector_add_format.cu --match-full-lines --input-file %T/vector_add_format.dp.cpp

#include <cuda.h>
#include <stdio.h>
#define VECTOR_SIZE 256

__global__ void VectorAddKernel(float* A, float* B, float* C)
{
    A[threadIdx.x] = threadIdx.x + 1.0f;
    B[threadIdx.x] = threadIdx.x + 1.0f;
    C[threadIdx.x] = A[threadIdx.x] + B[threadIdx.x];
}

int main()
{
    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, VECTOR_SIZE*sizeof(float));
    cudaMalloc(&d_B, VECTOR_SIZE*sizeof(float));
    cudaMalloc(&d_C, VECTOR_SIZE*sizeof(float));


     //CHECK:    dpct::get_default_queue_wait().submit([&](sycl::handler &cgh) {
//CHECK-NEXT:        cgh.parallel_for(
//CHECK-NEXT:            sycl::nd_range<3>(sycl::range<3>(1, 1, 1) *
//CHECK-NEXT:                                  sycl::range<3>(1, 1, VECTOR_SIZE),
//CHECK-NEXT:                              sycl::range<3>(1, 1, VECTOR_SIZE)),
//CHECK-NEXT:            [=](sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:                VectorAddKernel(d_A, d_B, d_C, item_ct1);
//CHECK-NEXT:            });
//CHECK-NEXT:    });
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