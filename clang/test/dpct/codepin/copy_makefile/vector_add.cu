
// RUN: dpct -out-root %T/vector_add %s --cuda-include-path="%cuda-path/include" --enable-codepin --gen-build-script -- -std=c++14  -x cuda --cuda-host-only
// RUN: FileCheck -strict-whitespace %s --match-full-lines --input-file %T/vector_add_codepin_sycl/vector_add.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/vector_add_codepin_sycl/vector_add.dp.cpp -o %T/vector_add_codepin_sycl/vector_add.dp.o %}
// RUN: cd %T/vector_add_codepin_cuda
// RUN: ls > default.log
// RUN: FileCheck --input-file default.log --match-full-lines %T/vector_add_codepin_sycl/vector_add.dp.cpp -check-prefix=DEFAULT
// DEFAULT: Makefile
#include <cuda.h>
#include <stdio.h>
#define VECTOR_SIZE 256

//     CHECK:void VectorAddKernel(float* A, float* B, float* C,
// CHECK-NEXT:                     const sycl::nd_item<3> &item_ct1)
//CHECK-NEXT:{
//CHECK-NEXT:    A[item_ct1.get_local_id(2)] = item_ct1.get_local_id(2) + 1.0f;
//CHECK-NEXT:    B[item_ct1.get_local_id(2)] = item_ct1.get_local_id(2) + 1.0f;
//CHECK-NEXT:    C[item_ct1.get_local_id(2)] = A[item_ct1.get_local_id(2)] + B[item_ct1.get_local_id(2)];
//CHECK-NEXT:}
__global__ void VectorAddKernel(float* A, float* B, float* C)
{
    A[threadIdx.x] = threadIdx.x + 1.0f;
    B[threadIdx.x] = threadIdx.x + 1.0f;
    C[threadIdx.x] = A[threadIdx.x] + B[threadIdx.x];
}



int main()
{
  float *d_A, *d_B, *d_C;

  cudaMalloc(&d_A, VECTOR_SIZE * sizeof(float));
  cudaMalloc(&d_B, VECTOR_SIZE * sizeof(float));
  cudaMalloc(&d_C, VECTOR_SIZE * sizeof(float));


  VectorAddKernel<<<1, VECTOR_SIZE>>>(d_A, d_B, d_C);
  float Result[VECTOR_SIZE] = {};
  cudaMemcpy(Result, d_C, VECTOR_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

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
