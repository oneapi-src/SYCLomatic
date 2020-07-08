#include <cublas_v2.h>
#include <curand.h>
#include <cuda_runtime.h>
// CHECK: #define ATOMIC_UPDATE( x ) dpct::atomic_fetch_add( &x, (unsigned int)1 );
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
  // CHECK: {
  // CHECK-NEXT: auto d_A_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_A_S);
  // CHECK-NEXT: auto d_B_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_B_S);
  // CHECK-NEXT: auto d_C_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_C_S);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (oneapi::mkl::blas::gemm(*handle, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, N, N, N, dpct::get_value(&alpha_S, *handle), d_A_S_buf_ct{{[0-9]+}}, N, d_B_S_buf_ct{{[0-9]+}}, N, dpct::get_value(&beta_S, *handle), d_C_S_buf_ct{{[0-9]+}}, N), 0);
  // CHECK-NEXT: }
  status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);
}

// CHECK: void randomGen(){
// CHECK-NEXT:   oneapi::mkl::rng::uniform<float> distr_ct{{[0-9]+}};
// CHECK-NEXT:   oneapi::mkl::rng::philox4x32x10* rng;
// CHECK-NEXT:   rng = new oneapi::mkl::rng::philox4x32x10(dpct::get_default_queue(), 1337ull);
// CHECK-NEXT:   /*
// CHECK-NEXT:   DPCT1026:{{[0-9]+}}: The call to curandSetPseudoRandomGeneratorSeed was removed, because the function call is redundant in DPC++.
// CHECK-NEXT:   */
// CHECK-NEXT:   float *d_data;
// CHECK-NEXT:   {
// CHECK-NEXT:   auto d_data_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_data);
// CHECK-NEXT:   oneapi::mkl::rng::generate(distr_ct{{[0-9]+}}, *rng, 100*100, d_data_buf_ct4);
// CHECK-NEXT:   }
// CHECK-NEXT: }
void randomGen(){
  curandGenerator_t rng;
  curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10);
  curandSetPseudoRandomGeneratorSeed(rng, 1337ull);
  float *d_data;
  curandGenerateUniform(rng, d_data, 100*100);
}
