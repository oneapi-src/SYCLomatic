// RUN: echo "test"
#define test_x 0

#include <cuda.h>

#define VECTOR_SIZE 256

__global__ void VectorAddKernel(float *A, float *B, float *C) {
  A[threadIdx.x] = threadIdx.x + 1.0f;
  B[threadIdx.x] = threadIdx.x + 1.0f;
  C[threadIdx.x] = A[threadIdx.x] + B[threadIdx.x];
}
#define CALL(x) if(0!=x){int a=4;}

int main() {
  float *d_A, *d_B, *d_C;
  cudaError_t status;

  //CHECK: dpctexp::codepin::get_ptr_size_map()[d_A] = VECTOR_SIZE * sizeof(float);
  CALL(cudaMalloc(&d_A, VECTOR_SIZE * sizeof(float)));
  //CHECK: dpctexp::codepin::get_ptr_size_map()[d_B] = VECTOR_SIZE * sizeof(float);
  cudaMalloc(&d_B, VECTOR_SIZE * sizeof(float));
  //CHECK: dpctexp::codepin::get_ptr_size_map()[d_C] = VECTOR_SIZE * sizeof(float);
  cudaMalloc(&d_C, VECTOR_SIZE * sizeof(float));
  //CHECK: dpctexp::codepin::gen_prolog_API_CP("{{[._0-9a-zA-Z\/\(\)\:\-]+}}", &q_ct1, "d_A", d_A, "d_B", d_B, "d_C", d_C);
  VectorAddKernel<<<1, VECTOR_SIZE>>>(d_A, d_B, d_C);
  //CHECK: dpctexp::codepin::gen_epilog_API_CP("{{[._0-9a-zA-Z\/\(\)\:\-]+}}", &q_ct1, "d_A", d_A, "d_B", d_B, "d_C", d_C);
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
