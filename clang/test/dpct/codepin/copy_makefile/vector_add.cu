// UNSUPPORTED: system-windows
// RUN: dpct -out-root %T/vector_add %s --cuda-include-path="%cuda-path/include" --enable-codepin --gen-build-script -- -std=c++14  -x cuda --cuda-host-only
// RUN: echo "begin" > %T/diff_sycl_makefile.txt
// RUN: diff --strip-trailing-cr %S/expected_makefile %T/vector_add_codepin_sycl/Makefile.dpct >> %T/diff_sycl_makefile.txt
// RUN: echo "end" >> %T/diff_sycl_makefile.txt
// RUN: FileCheck --input-file %T/diff_sycl_makefile.txt --check-prefix=CHECK %s

// RUN: cd %T/vector_add_codepin_cuda
// RUN: ls > default.log
// RUN: FileCheck --input-file default.log --match-full-lines %T/vector_add_codepin_sycl/vector_add.dp.cpp -check-prefix=DEFAULT
// DEFAULT: Makefile

// RUN: echo "begin" > %T/diff.txt
// RUN: diff --strip-trailing-cr %S/expected.txt %T/vector_add_codepin_cuda/Makefile >> %T/diff.txt
// RUN: echo "end" >> %T/diff.txt
// RUN: FileCheck --input-file %T/diff.txt --check-prefix=CHECK %s
// CHECK: begin
// CHECK-NEXT:end
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
