#include <cublas_v2.h>
#include <cuda_runtime.h>
// CHECK: #define ATOMIC_UPDATE( x ) syclct::atomic_fetch_add( &x, (unsigned int)(1) );
#define ATOMIC_UPDATE( x ) atomicAdd( &x, 1 );

// CHECK: int global_id(cl::sycl::nd_item<3> item_{{[a-f0-9]+}});
__device__ int global_id();


// CHECK: void simple_kernel(unsigned *i_array, cl::sycl::nd_item<3> item_{{[a-f0-9]+}}) {
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
  // CHECK-NEXT: auto d_A_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(d_A_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> d_A_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = d_A_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(d_A_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto d_B_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(d_B_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> d_B_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = d_B_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(d_B_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto d_C_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(d_C_S);
  // CHECK-NEXT: cl::sycl::buffer<float,1> d_C_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = d_C_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(d_C_S_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  // CHECK-NEXT: status = (mkl::sgemm(handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, *(&alpha_S), d_A_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, N, d_B_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, N, *(&beta_S), d_C_S_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, N), 0);
  // CHECK-NEXT: }
  status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);
}
