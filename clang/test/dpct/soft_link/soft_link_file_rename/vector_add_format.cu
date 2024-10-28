// UNSUPPORTED: system-windows
// RUN: cp -r %S %T && cd %T/soft_link_file_rename/link/test &&  rm test.cuh && ln -nfs  ../../target/test/test.cuh test.cuh
// RUN: dpct  --in-root=%T/soft_link_file_rename --out-root=%T/out  --cuda-include-path="%cuda-path/include" --process-all -- -I %T/soft_link_file_rename/link -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/out/link/test/test.dp.hpp --match-full-lines %T/soft_link_file_rename/link/test/test.cuh
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST  %T/out/vector_add_format.dp.cpp -o %T/out/vector_add_format.dp.o -I %T/out/link %}

#include <cuda.h>
#include <stdio.h>
#include "test/test.cuh"
#define VECTOR_SIZE 256

__global__ void VectorAddKernel(float* A, float* B, float* C)
{
    A[threadIdx.x] = threadIdx.x + 1.0f;
    B[threadIdx.x] = threadIdx.x + 1.0f;
    C[threadIdx.x] = A[threadIdx.x] + B[threadIdx.x];
}



int main()
{
  //      CHECK:    dpct::device_ext &dev_ct1 = dpct::get_current_device();
  // CHECK-NEXT:    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  float *d_A, *d_B, *d_C;


  cudaMalloc(&d_A, VECTOR_SIZE * sizeof(float));
  cudaMalloc(&d_B, VECTOR_SIZE * sizeof(float));
  cudaMalloc(&d_C, VECTOR_SIZE * sizeof(float));


  VectorAddKernel<<<1, VECTOR_SIZE>>>(d_A, d_B, d_C);


  float Result[VECTOR_SIZE] = {};
  cudaMemcpy(Result, d_C, VECTOR_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

  //      CHECK:  dpct::dpct_free(d_A, q_ct1);
  // CHECK-NEXT:  dpct::dpct_free(d_B, q_ct1);
  // CHECK-NEXT:  dpct::dpct_free(d_C, q_ct1);
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
