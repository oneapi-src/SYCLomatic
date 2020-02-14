#include <cublas_v2.h>
#include <cuda_runtime.h>
// CHECK: #define ATOMIC_UPDATE( x ) dpct::atomic_fetch_add( &x, (unsigned int)(1) );
#define ATOMIC_UPDATE( x ) atomicAdd( &x, 1 );

// CHECK: int global_id(sycl::nd_item<3> item_ct1);
__device__ int global_id();


// CHECK: void simple_kernel(unsigned *i_array, sycl::nd_item<3> item_ct1) {
__global__ void simple_kernel(unsigned *i_array) {
  int index;
  index = global_id();
  if (index < 360) {
    i_array[index] = index;
    ATOMIC_UPDATE(i_array[index])
  }
  return;
}

void sgemm() {
  cublasStatus_t status;
  cublasHandle_t handle;
  int N = 275;
  float *d_A_S = 0;
  float *d_B_S = 0;
  float *d_C_S = 0;
  float alpha_S = 1.0f;
  float beta_S = 0.0f;
  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto d_A_S_buff_ct1 = dpct::get_buffer<float>(d_A_S);
  // CHECK-NEXT: auto d_B_S_buff_ct1 = dpct::get_buffer<float>(d_B_S);
  // CHECK-NEXT: auto d_C_S_buff_ct1 = dpct::get_buffer<float>(d_C_S);
  // CHECK-NEXT: status = (mkl::blas::gemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, *(&alpha_S), d_A_S_buff_ct1, N, d_B_S_buff_ct1, N, *(&beta_S), d_C_S_buff_ct1, N), 0);
  // CHECK-NEXT: }
  status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);
}
