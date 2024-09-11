// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --helper-function-preference=no-queue-device --optimize-migration --enable-profiling --use-syclcompat --format-range=none --out-root %T/syclcompat_test1 %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --input-file %T/syclcompat_test1/syclcompat_test1.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -DBUILD_TEST -c -fsycl %T/syclcompat_test1/syclcompat_test1.dp.cpp -o %T/syclcompat_test1/syclcompat_test1.dp.o %}

// CHECK: #include <oneapi/dpl/execution>
// CHECK-NEXT: #include <oneapi/dpl/algorithm>
// CHECK-NEXT: #define SYCLCOMPAT_PROFILING_ENABLED
// CHECK-NEXT: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <syclcompat/syclcompat.hpp>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

// CHECK: __syclcompat_inline__ void kernel() {}
__global__ void kernel() {}

void f1() {
  float *f;
  cudaEvent_t start;
  // CHECK: syclcompat::err0 err = SYCLCOMPAT_CHECK_ERROR(f = sycl::malloc_device<float>(1, q_ct1));
  cudaError_t err = cudaMalloc(&f, sizeof(float));
  // CHECK: /*
  // CHECK-NEXT: DPCT1131:{{[0-9]+}}: The migration of "cudaEventRecord" is not currently supported with SYCLcompat. Please adjust the code manually.
  // CHECK-NEXT: */
#ifndef BUILD_TEST
  cudaEventRecord(start, 0);
#endif
  cudaDeviceProp deviceProp;
  // CHECK: /*
  // CHECK-NEXT: DPCT1131:{{[0-9]+}}: The migration of "{{cudaGetDeviceProperties(_v2)?}}" is not currently supported with SYCLcompat. Please adjust the code manually.
  // CHECK-NEXT: */
#ifndef BUILD_TEST
  cudaGetDeviceProperties(&deviceProp, 0);
#endif
  // CHECK: syclcompat::dim3 block(8, 1, 1);
  // CHECK-NEXT: q_ct1.parallel_for(
  // CHECK-NEXT:   sycl::nd_range<3>(sycl::range<3>(1, 1, 8) * static_cast<sycl::range<3>>(block), static_cast<sycl::range<3>>(block)), 
  // CHECK-NEXT:   [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:     kernel();
  // CHECK-NEXT:   });
  dim3 block(8, 1, 1);
  kernel<<<8, block>>>();
}

void f2() {
  cusolverDnHandle_t* cusolverH;;
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
  int n = 0;
  int nrhs = 0;
  float B_f = 0;
  float C_f = 0;
  int lda = 0;
  int ldb = 0;
  int devInfo = 0;
  // CHECK: {
  // CHECK-NEXT: std::int64_t scratchpad_size_ct1 = oneapi::mkl::lapack::potrs_scratchpad_size<float>(**cusolverH ,uplo ,n ,nrhs ,lda ,ldb);
  // CHECK-NEXT: float *scratchpad_ct2 = sycl::malloc_device<float>(scratchpad_size_ct1, **cusolverH);
  // CHECK-NEXT: sycl::event event_ct3;
  // CHECK-NEXT: event_ct3 = oneapi::mkl::lapack::potrs(**cusolverH, uplo, n, nrhs, (float*)&C_f, lda, (float*)&B_f, ldb, scratchpad_ct2, scratchpad_size_ct1);
  // CHECK-NEXT: std::vector<void *> ws_vec_ct4{scratchpad_ct2};
  // CHECK-NEXT: syclcompat::enqueue_free(ws_vec_ct4, {event_ct3}, **cusolverH);
  // CHECK-NEXT: }
  cusolverDnSpotrs(*cusolverH, uplo, n, nrhs, &C_f, lda, &B_f, ldb, &devInfo);
}

void f3() {
  cublasHandle_t handle;
  int N = 275;
  float alpha_S = 1.0f;
  float beta_S = 1.0f;
  int trans0 = 0;
  const float *x_S = 0;
  const float *y_S = 0;
  float *result_S = 0;
  int lda = 1;
  int incx = 1;
  int incy = 1;
  // CHECK: oneapi::mkl::blas::column_major::gemv(handle->get_queue(), dpct::get_transpose(trans0), N, N, alpha_S, x_S, lda, y_S, incx, beta_S, result_S, incy);
  cublasSgemv(handle, (cublasOperation_t)trans0, N, N, &alpha_S, x_S, lda, y_S, incx, &beta_S, result_S, incy);
}

void f4() {
  // CHECK: dpct::device_vector<int> d_v(10);
  // CHECK-NEXT: dpct::device_pointer<int> d_ptr = d_v.data();
  thrust::device_vector<int> d_v(10);
  thrust::device_ptr<int> d_ptr = d_v.data();
}
