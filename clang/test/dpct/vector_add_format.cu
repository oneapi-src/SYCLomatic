// RUN: mkdir %T/vector_add_format
// RUN: cat %s > %T/vector_add_format/vector_add_format.cu
// RUN: cd %T/vector_add_format
// RUN: dpct -out-root %T/vector_add_format vector_add_format.cu --cuda-include-path="%cuda-path/include" -- -std=c++14  -x cuda --cuda-host-only
// RUN: FileCheck -strict-whitespace %s --match-full-lines --input-file %T/vector_add_format/vector_add_format.dp.cpp
// RUN: cd ..
// RUN: rm -rf ./vector_add_format

#include <cuda.h>
#include <stdio.h>
#define VECTOR_SIZE 256

//     CHECK:void VectorAddKernel(float* A, float* B, float* C, sycl::nd_item<3> item_ct1)
//CHECK-NEXT:{
//CHECK-NEXT:    A[item_ct1.get_local_id(2)] = item_ct1.get_local_id(2) + 1.0f;
//CHECK-NEXT:    B[item_ct1.get_local_id(2)] = item_ct1.get_local_id(2) + 1.0f;
//CHECK-NEXT:    C[item_ct1.get_local_id(2)] =
//CHECK-NEXT:        A[item_ct1.get_local_id(2)] + B[item_ct1.get_local_id(2)];
//CHECK-NEXT:}
__global__ void VectorAddKernel(float* A, float* B, float* C)
{
    A[threadIdx.x] = threadIdx.x + 1.0f;
    B[threadIdx.x] = threadIdx.x + 1.0f;
    C[threadIdx.x] = A[threadIdx.x] + B[threadIdx.x];
}



int main()
{
//     CHECK:    dpct::device_ext &dev_ct1 = dpct::get_current_device();
//CHECK-NEXT:    sycl::queue &q_ct1 = dev_ct1.default_queue();
    float *d_A, *d_B, *d_C;

//     CHECK:    d_A = sycl::malloc_device<float>(VECTOR_SIZE, q_ct1);
//CHECK-NEXT:    d_B = sycl::malloc_device<float>(VECTOR_SIZE, q_ct1);
//CHECK-NEXT:    d_C = sycl::malloc_device<float>(VECTOR_SIZE, q_ct1);
    cudaMalloc(&d_A, VECTOR_SIZE*sizeof(float));
    cudaMalloc(&d_B, VECTOR_SIZE*sizeof(float));
    cudaMalloc(&d_C, VECTOR_SIZE*sizeof(float));


//     CHECK:    q_ct1.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, VECTOR_SIZE),
//CHECK-NEXT:                                         sycl::range<3>(1, 1, VECTOR_SIZE)),
//CHECK-NEXT:                       [=](sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:                           VectorAddKernel(d_A, d_B, d_C, item_ct1);
//CHECK-NEXT:                       });
    VectorAddKernel<<<1, VECTOR_SIZE>>>(d_A, d_B, d_C);

//     CHECK:    float Result[VECTOR_SIZE] = { };
//CHECK-NEXT:    q_ct1.memcpy(Result, d_C, VECTOR_SIZE * sizeof(float)).wait();
    float Result[VECTOR_SIZE] = { };
    cudaMemcpy(Result, d_C, VECTOR_SIZE*sizeof(float), cudaMemcpyDeviceToHost);

//     CHECK:    sycl::free(d_A, q_ct1);
//CHECK-NEXT:    sycl::free(d_B, q_ct1);
//CHECK-NEXT:    sycl::free(d_C, q_ct1);
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