// UNSUPPORTED: system-windows
// RUN: cp -r %S %T && cd %T/soft_link_mul_dir2/link/hello && rm -rf target_soft_link && ln -nfs  ../../target target_soft_link && cd %T/soft_link_mul_dir2/link && rm -rf link && ln -nfs  hello link
// RUN: dpct  --in-root=%T/soft_link_mul_dir2 --out-root=%T/out  --cuda-include-path="%cuda-path/include" --process-all -- -I %T/soft_link_mul_dir2/link/link/target_soft_link -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/out/link/link/target_soft_link/test/test.hpp --match-full-lines %T/soft_link_mul_dir2/target/test/test.hpp
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST  %T/out/vector_add_format.dp.cpp -o %T/out/vector_add_format.dp.o -I %T/out/link/hello/target_soft_link %}

#include <cuda.h>
#include <stdio.h>
#include "test/test.hpp"
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
