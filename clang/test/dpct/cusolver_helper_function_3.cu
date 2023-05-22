// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0
// RUN: dpct --format-range=none -out-root %T/cusolver_helper_function_3 %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cusolver_helper_function_3/cusolver_helper_function_3.dp.cpp --match-full-lines %s

//CHECK:#include <sycl/sycl.hpp>
//CHECK-NEXT:#include <dpct/dpct.hpp>
//CHECK-NEXT:#include <dpct/lapack_utils.hpp>
#include "cusolverDn.h"

void foo1() {
  cusolverStatus_t status;

  float* a_s;
  double* a_d;
  float2* a_c;
  double2* a_z;
  int64_t* tau_s;
  int64_t* tau_d;
  int64_t* tau_c;
  int64_t* tau_z;

  //CHECK:sycl::queue* handle;
  //CHECK-NEXT:status = CHECK_SYCL_ERROR(handle = &dpct::get_default_queue());
  cusolverDnHandle_t handle;
  status = cusolverDnCreate(&handle);

  size_t device_ws_size_s;
  size_t device_ws_size_d;
  size_t device_ws_size_c;
  size_t device_ws_size_z;
  size_t host_ws_size_s;
  size_t host_ws_size_d;
  size_t host_ws_size_c;
  size_t host_ws_size_z;

  //CHECK:int params;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1027:{{[0-9]+}}: The call to cusolverDnCreateParams was replaced with 0 because this call is redundant in SYCL.
  //CHECK-NEXT:*/
  //CHECK-NEXT:status = 0;
  cusolverDnParams_t params;
  status = cusolverDnCreateParams(&params);

  //CHECK:status = dpct::lapack::geqrf_scratchpad_size(*handle, 2, 2, dpct::library_data_t::real_float, 2, &device_ws_size_s, &host_ws_size_s);
  //CHECK-NEXT:status = dpct::lapack::geqrf_scratchpad_size(*handle, 2, 2, dpct::library_data_t::real_double, 2, &device_ws_size_d, &host_ws_size_d);
  //CHECK-NEXT:status = dpct::lapack::geqrf_scratchpad_size(*handle, 2, 2, dpct::library_data_t::complex_float, 2, &device_ws_size_c, &host_ws_size_c);
  //CHECK-NEXT:status = dpct::lapack::geqrf_scratchpad_size(*handle, 2, 2, dpct::library_data_t::complex_double, 2, &device_ws_size_z, &host_ws_size_z);
  status = cusolverDnXgeqrf_bufferSize(handle, params, 2, 2, CUDA_R_32F, a_s, 2, CUDA_R_32F, tau_s, CUDA_R_32F, &device_ws_size_s, &host_ws_size_s);
  status = cusolverDnXgeqrf_bufferSize(handle, params, 2, 2, CUDA_R_64F, a_d, 2, CUDA_R_64F, tau_d, CUDA_R_64F, &device_ws_size_d, &host_ws_size_d);
  status = cusolverDnXgeqrf_bufferSize(handle, params, 2, 2, CUDA_C_32F, a_c, 2, CUDA_C_32F, tau_c, CUDA_C_32F, &device_ws_size_c, &host_ws_size_c);
  status = cusolverDnXgeqrf_bufferSize(handle, params, 2, 2, CUDA_C_64F, a_z, 2, CUDA_C_64F, tau_z, CUDA_C_64F, &device_ws_size_z, &host_ws_size_z);

  void* device_ws_s;
  void* device_ws_d;
  void* device_ws_c;
  void* device_ws_z;
  void* host_ws_s;
  void* host_ws_d;
  void* host_ws_c;
  void* host_ws_z;

  int *info;

  //CHECK:status = dpct::lapack::geqrf(*handle, 2, 2, dpct::library_data_t::real_float, a_s, 2, dpct::library_data_t::real_float, tau_s, device_ws_s, device_ws_size_s, info);
  //CHECK-NEXT:status = dpct::lapack::geqrf(*handle, 2, 2, dpct::library_data_t::real_double, a_d, 2, dpct::library_data_t::real_double, tau_d, device_ws_d, device_ws_size_d, info);
  //CHECK-NEXT:status = dpct::lapack::geqrf(*handle, 2, 2, dpct::library_data_t::complex_float, a_c, 2, dpct::library_data_t::complex_float, tau_c, device_ws_c, device_ws_size_c, info);
  //CHECK-NEXT:status = dpct::lapack::geqrf(*handle, 2, 2, dpct::library_data_t::complex_double, a_z, 2, dpct::library_data_t::complex_double, tau_z, device_ws_z, device_ws_size_z, info);
  status = cusolverDnXgeqrf(handle, params, 2, 2, CUDA_R_32F, a_s, 2, CUDA_R_32F, tau_s, CUDA_R_32F, device_ws_s, device_ws_size_s, host_ws_s, host_ws_size_s, info);
  status = cusolverDnXgeqrf(handle, params, 2, 2, CUDA_R_64F, a_d, 2, CUDA_R_64F, tau_d, CUDA_R_64F, device_ws_d, device_ws_size_d, host_ws_d, host_ws_size_d, info);
  status = cusolverDnXgeqrf(handle, params, 2, 2, CUDA_C_32F, a_c, 2, CUDA_C_32F, tau_c, CUDA_C_32F, device_ws_c, device_ws_size_c, host_ws_c, host_ws_size_c, info);
  status = cusolverDnXgeqrf(handle, params, 2, 2, CUDA_C_64F, a_z, 2, CUDA_C_64F, tau_z, CUDA_C_64F, device_ws_z, device_ws_size_z, host_ws_z, host_ws_size_z, info);

  //CHECK:/*
  //CHECK-NEXT:DPCT1027:{{[0-9]+}}: The call to cusolverDnDestroyParams was replaced with 0 because this call is redundant in SYCL.
  //CHECK-NEXT:*/
  //CHECK-NEXT:status = 0;
  //CHECK-NEXT:status = CHECK_SYCL_ERROR(handle = nullptr);
  status = cusolverDnDestroyParams(params);
  status = cusolverDnDestroy(handle);
}

void foo2() {
  cusolverStatus_t status;

  float* a_s;
  double* a_d;
  float2* a_c;
  double2* a_z;
  int64_t* tau_s;
  int64_t* tau_d;
  int64_t* tau_c;
  int64_t* tau_z;

  //CHECK:sycl::queue* handle;
  //CHECK-NEXT:status = CHECK_SYCL_ERROR(handle = &dpct::get_default_queue());
  cusolverDnHandle_t handle;
  status = cusolverDnCreate(&handle);

  size_t device_ws_size_s;
  size_t device_ws_size_d;
  size_t device_ws_size_c;
  size_t device_ws_size_z;
  size_t host_ws_size_s;
  size_t host_ws_size_d;
  size_t host_ws_size_c;
  size_t host_ws_size_z;

  //CHECK:int params;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1027:{{[0-9]+}}: The call to cusolverDnCreateParams was replaced with 0 because this call is redundant in SYCL.
  //CHECK-NEXT:*/
  //CHECK-NEXT:status = 0;
  cusolverDnParams_t params;
  status = cusolverDnCreateParams(&params);

  //CHECK:status = dpct::lapack::geqrf_scratchpad_size(*handle, 2, 2, dpct::library_data_t::real_float, 2, &device_ws_size_s);
  //CHECK-NEXT:status = dpct::lapack::geqrf_scratchpad_size(*handle, 2, 2, dpct::library_data_t::real_double, 2, &device_ws_size_d);
  //CHECK-NEXT:status = dpct::lapack::geqrf_scratchpad_size(*handle, 2, 2, dpct::library_data_t::complex_float, 2, &device_ws_size_c);
  //CHECK-NEXT:status = dpct::lapack::geqrf_scratchpad_size(*handle, 2, 2, dpct::library_data_t::complex_double, 2, &device_ws_size_z);
  status = cusolverDnGeqrf_bufferSize(handle, params, 2, 2, CUDA_R_32F, a_s, 2, CUDA_R_32F, tau_s, CUDA_R_32F, &device_ws_size_s);
  status = cusolverDnGeqrf_bufferSize(handle, params, 2, 2, CUDA_R_64F, a_d, 2, CUDA_R_64F, tau_d, CUDA_R_64F, &device_ws_size_d);
  status = cusolverDnGeqrf_bufferSize(handle, params, 2, 2, CUDA_C_32F, a_c, 2, CUDA_C_32F, tau_c, CUDA_C_32F, &device_ws_size_c);
  status = cusolverDnGeqrf_bufferSize(handle, params, 2, 2, CUDA_C_64F, a_z, 2, CUDA_C_64F, tau_z, CUDA_C_64F, &device_ws_size_z);

  void* device_ws_s;
  void* device_ws_d;
  void* device_ws_c;
  void* device_ws_z;
  void* host_ws_s;
  void* host_ws_d;
  void* host_ws_c;
  void* host_ws_z;

  int *info;

  //CHECK:status = dpct::lapack::geqrf(*handle, 2, 2, dpct::library_data_t::real_float, a_s, 2, dpct::library_data_t::real_float, tau_s, device_ws_s, device_ws_size_s, info);
  //CHECK-NEXT:status = dpct::lapack::geqrf(*handle, 2, 2, dpct::library_data_t::real_double, a_d, 2, dpct::library_data_t::real_double, tau_d, device_ws_d, device_ws_size_d, info);
  //CHECK-NEXT:status = dpct::lapack::geqrf(*handle, 2, 2, dpct::library_data_t::complex_float, a_c, 2, dpct::library_data_t::complex_float, tau_c, device_ws_c, device_ws_size_c, info);
  //CHECK-NEXT:status = dpct::lapack::geqrf(*handle, 2, 2, dpct::library_data_t::complex_double, a_z, 2, dpct::library_data_t::complex_double, tau_z, device_ws_z, device_ws_size_z, info);
  status = cusolverDnGeqrf(handle, params, 2, 2, CUDA_R_32F, a_s, 2, CUDA_R_32F, tau_s, CUDA_R_32F, device_ws_s, device_ws_size_s, info);
  status = cusolverDnGeqrf(handle, params, 2, 2, CUDA_R_64F, a_d, 2, CUDA_R_64F, tau_d, CUDA_R_64F, device_ws_d, device_ws_size_d, info);
  status = cusolverDnGeqrf(handle, params, 2, 2, CUDA_C_32F, a_c, 2, CUDA_C_32F, tau_c, CUDA_C_32F, device_ws_c, device_ws_size_c, info);
  status = cusolverDnGeqrf(handle, params, 2, 2, CUDA_C_64F, a_z, 2, CUDA_C_64F, tau_z, CUDA_C_64F, device_ws_z, device_ws_size_z, info);

  //CHECK:/*
  //CHECK-NEXT:DPCT1027:{{[0-9]+}}: The call to cusolverDnDestroyParams was replaced with 0 because this call is redundant in SYCL.
  //CHECK-NEXT:*/
  //CHECK-NEXT:status = 0;
  //CHECK-NEXT:status = CHECK_SYCL_ERROR(handle = nullptr);
  status = cusolverDnDestroyParams(params);
  status = cusolverDnDestroy(handle);
}

void foo3() {
  cusolverStatus_t status;

  float* a_s;
  double* a_d;
  float2* a_c;
  double2* a_z;
  int64_t* ipiv_s;
  int64_t* ipiv_d;
  int64_t* ipiv_c;
  int64_t* ipiv_z;

  //CHECK:sycl::queue* handle;
  //CHECK-NEXT:status = CHECK_SYCL_ERROR(handle = &dpct::get_default_queue());
  cusolverDnHandle_t handle;
  status = cusolverDnCreate(&handle);

  size_t device_ws_size_s;
  size_t device_ws_size_d;
  size_t device_ws_size_c;
  size_t device_ws_size_z;
  size_t host_ws_size_s;
  size_t host_ws_size_d;
  size_t host_ws_size_c;
  size_t host_ws_size_z;

  //CHECK:int params;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1027:{{[0-9]+}}: The call to cusolverDnCreateParams was replaced with 0 because this call is redundant in SYCL.
  //CHECK-NEXT:*/
  //CHECK-NEXT:status = 0;
  cusolverDnParams_t params;
  status = cusolverDnCreateParams(&params);

  //CHECK:status = dpct::lapack::getrf_scratchpad_size(*handle, 2, 2, dpct::library_data_t::real_float, 2, &device_ws_size_s, &host_ws_size_s);
  //CHECK-NEXT:status = dpct::lapack::getrf_scratchpad_size(*handle, 2, 2, dpct::library_data_t::real_double, 2, &device_ws_size_d, &host_ws_size_d);
  //CHECK-NEXT:status = dpct::lapack::getrf_scratchpad_size(*handle, 2, 2, dpct::library_data_t::complex_float, 2, &device_ws_size_c, &host_ws_size_c);
  //CHECK-NEXT:status = dpct::lapack::getrf_scratchpad_size(*handle, 2, 2, dpct::library_data_t::complex_double, 2, &device_ws_size_z, &host_ws_size_z);
  status = cusolverDnXgetrf_bufferSize(handle, params, 2, 2, CUDA_R_32F, a_s, 2, CUDA_R_32F, &device_ws_size_s, &host_ws_size_s);
  status = cusolverDnXgetrf_bufferSize(handle, params, 2, 2, CUDA_R_64F, a_d, 2, CUDA_R_64F, &device_ws_size_d, &host_ws_size_d);
  status = cusolverDnXgetrf_bufferSize(handle, params, 2, 2, CUDA_C_32F, a_c, 2, CUDA_C_32F, &device_ws_size_c, &host_ws_size_c);
  status = cusolverDnXgetrf_bufferSize(handle, params, 2, 2, CUDA_C_64F, a_z, 2, CUDA_C_64F, &device_ws_size_z, &host_ws_size_z);

  void* device_ws_s;
  void* device_ws_d;
  void* device_ws_c;
  void* device_ws_z;
  void* host_ws_s;
  void* host_ws_d;
  void* host_ws_c;
  void* host_ws_z;

  int *info;

  //CHECK:/*
  //CHECK-NEXT:DPCT1047:{{[0-9]+}}: The meaning of ipiv_s in the dpct::lapack::getrf is different from the cusolverDnXgetrf. You may need to check the migrated code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:status = dpct::lapack::getrf(*handle, 2, 2, dpct::library_data_t::real_float, a_s, 2, ipiv_s, device_ws_s, device_ws_size_s, info);
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1047:{{[0-9]+}}: The meaning of ipiv_d in the dpct::lapack::getrf is different from the cusolverDnXgetrf. You may need to check the migrated code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:status = dpct::lapack::getrf(*handle, 2, 2, dpct::library_data_t::real_double, a_d, 2, ipiv_d, device_ws_d, device_ws_size_d, info);
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1047:{{[0-9]+}}: The meaning of ipiv_c in the dpct::lapack::getrf is different from the cusolverDnXgetrf. You may need to check the migrated code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:status = dpct::lapack::getrf(*handle, 2, 2, dpct::library_data_t::complex_float, a_c, 2, ipiv_c, device_ws_c, device_ws_size_c, info);
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1047:{{[0-9]+}}: The meaning of ipiv_z in the dpct::lapack::getrf is different from the cusolverDnXgetrf. You may need to check the migrated code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:status = dpct::lapack::getrf(*handle, 2, 2, dpct::library_data_t::complex_double, a_z, 2, ipiv_z, device_ws_z, device_ws_size_z, info);
  status = cusolverDnXgetrf(handle, params, 2, 2, CUDA_R_32F, a_s, 2, ipiv_s, CUDA_R_32F, device_ws_s, device_ws_size_s, host_ws_s, host_ws_size_s, info);
  status = cusolverDnXgetrf(handle, params, 2, 2, CUDA_R_64F, a_d, 2, ipiv_d, CUDA_R_64F, device_ws_d, device_ws_size_d, host_ws_d, host_ws_size_d, info);
  status = cusolverDnXgetrf(handle, params, 2, 2, CUDA_C_32F, a_c, 2, ipiv_c, CUDA_C_32F, device_ws_c, device_ws_size_c, host_ws_c, host_ws_size_c, info);
  status = cusolverDnXgetrf(handle, params, 2, 2, CUDA_C_64F, a_z, 2, ipiv_z, CUDA_C_64F, device_ws_z, device_ws_size_z, host_ws_z, host_ws_size_z, info);

  //CHECK:/*
  //CHECK-NEXT:DPCT1027:{{[0-9]+}}: The call to cusolverDnDestroyParams was replaced with 0 because this call is redundant in SYCL.
  //CHECK-NEXT:*/
  //CHECK-NEXT:status = 0;
  //CHECK-NEXT:status = CHECK_SYCL_ERROR(handle = nullptr);
  status = cusolverDnDestroyParams(params);
  status = cusolverDnDestroy(handle);
}

void foo4() {
  cusolverStatus_t status;

  float* a_s;
  double* a_d;
  float2* a_c;
  double2* a_z;
  int64_t* ipiv_s;
  int64_t* ipiv_d;
  int64_t* ipiv_c;
  int64_t* ipiv_z;

  //CHECK:sycl::queue* handle;
  //CHECK-NEXT:status = CHECK_SYCL_ERROR(handle = &dpct::get_default_queue());
  cusolverDnHandle_t handle;
  status = cusolverDnCreate(&handle);

  size_t device_ws_size_s;
  size_t device_ws_size_d;
  size_t device_ws_size_c;
  size_t device_ws_size_z;

  //CHECK:int params;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1027:{{[0-9]+}}: The call to cusolverDnCreateParams was replaced with 0 because this call is redundant in SYCL.
  //CHECK-NEXT:*/
  //CHECK-NEXT:status = 0;
  cusolverDnParams_t params;
  status = cusolverDnCreateParams(&params);

  //CHECK:status = dpct::lapack::getrf_scratchpad_size(*handle, 2, 2, dpct::library_data_t::real_float, 2, &device_ws_size_s);
  //CHECK-NEXT:status = dpct::lapack::getrf_scratchpad_size(*handle, 2, 2, dpct::library_data_t::real_double, 2, &device_ws_size_d);
  //CHECK-NEXT:status = dpct::lapack::getrf_scratchpad_size(*handle, 2, 2, dpct::library_data_t::complex_float, 2, &device_ws_size_c);
  //CHECK-NEXT:status = dpct::lapack::getrf_scratchpad_size(*handle, 2, 2, dpct::library_data_t::complex_double, 2, &device_ws_size_z);
  status = cusolverDnGetrf_bufferSize(handle, params, 2, 2, CUDA_R_32F, a_s, 2, CUDA_R_32F, &device_ws_size_s);
  status = cusolverDnGetrf_bufferSize(handle, params, 2, 2, CUDA_R_64F, a_d, 2, CUDA_R_64F, &device_ws_size_d);
  status = cusolverDnGetrf_bufferSize(handle, params, 2, 2, CUDA_C_32F, a_c, 2, CUDA_C_32F, &device_ws_size_c);
  status = cusolverDnGetrf_bufferSize(handle, params, 2, 2, CUDA_C_64F, a_z, 2, CUDA_C_64F, &device_ws_size_z);

  void* device_ws_s;
  void* device_ws_d;
  void* device_ws_c;
  void* device_ws_z;
  void* host_ws_s;
  void* host_ws_d;
  void* host_ws_c;
  void* host_ws_z;

  int *info;

  //CHECK:/*
  //CHECK-NEXT:DPCT1047:{{[0-9]+}}: The meaning of ipiv_s in the dpct::lapack::getrf is different from the cusolverDnGetrf. You may need to check the migrated code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:status = dpct::lapack::getrf(*handle, 2, 2, dpct::library_data_t::real_float, a_s, 2, ipiv_s, device_ws_s, device_ws_size_s, info);
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1047:{{[0-9]+}}: The meaning of ipiv_d in the dpct::lapack::getrf is different from the cusolverDnGetrf. You may need to check the migrated code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:status = dpct::lapack::getrf(*handle, 2, 2, dpct::library_data_t::real_double, a_d, 2, ipiv_d, device_ws_d, device_ws_size_d, info);
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1047:{{[0-9]+}}: The meaning of ipiv_c in the dpct::lapack::getrf is different from the cusolverDnGetrf. You may need to check the migrated code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:status = dpct::lapack::getrf(*handle, 2, 2, dpct::library_data_t::complex_float, a_c, 2, ipiv_c, device_ws_c, device_ws_size_c, info);
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1047:{{[0-9]+}}: The meaning of ipiv_z in the dpct::lapack::getrf is different from the cusolverDnGetrf. You may need to check the migrated code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:status = dpct::lapack::getrf(*handle, 2, 2, dpct::library_data_t::complex_double, a_z, 2, ipiv_z, device_ws_z, device_ws_size_z, info);
  status = cusolverDnGetrf(handle, params, 2, 2, CUDA_R_32F, a_s, 2, ipiv_s, CUDA_R_32F, device_ws_s, device_ws_size_s, info);
  status = cusolverDnGetrf(handle, params, 2, 2, CUDA_R_64F, a_d, 2, ipiv_d, CUDA_R_64F, device_ws_d, device_ws_size_d, info);
  status = cusolverDnGetrf(handle, params, 2, 2, CUDA_C_32F, a_c, 2, ipiv_c, CUDA_C_32F, device_ws_c, device_ws_size_c, info);
  status = cusolverDnGetrf(handle, params, 2, 2, CUDA_C_64F, a_z, 2, ipiv_z, CUDA_C_64F, device_ws_z, device_ws_size_z, info);

  //CHECK:/*
  //CHECK-NEXT:DPCT1027:{{[0-9]+}}: The call to cusolverDnDestroyParams was replaced with 0 because this call is redundant in SYCL.
  //CHECK-NEXT:*/
  //CHECK-NEXT:status = 0;
  //CHECK-NEXT:status = CHECK_SYCL_ERROR(handle = nullptr);
  status = cusolverDnDestroyParams(params);
  status = cusolverDnDestroy(handle);
}

void foo5() {
  cusolverStatus_t status;

  float* a_s;
  double* a_d;
  float2* a_c;
  double2* a_z;
  int64_t* ipiv_s;
  int64_t* ipiv_d;
  int64_t* ipiv_c;
  int64_t* ipiv_z;
  float* b_s;
  double* b_d;
  float2* b_c;
  double2* b_z;

  //CHECK:sycl::queue* handle;
  //CHECK-NEXT:status = CHECK_SYCL_ERROR(handle = &dpct::get_default_queue());
  cusolverDnHandle_t handle;
  status = cusolverDnCreate(&handle);

  //CHECK:int params;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1027:{{[0-9]+}}: The call to cusolverDnCreateParams was replaced with 0 because this call is redundant in SYCL.
  //CHECK-NEXT:*/
  //CHECK-NEXT:status = 0;
  cusolverDnParams_t params;
  status = cusolverDnCreateParams(&params);

  int *info;

  //CHECK:status = dpct::lapack::getrs(*handle, oneapi::mkl::transpose::nontrans, 2, 3, dpct::library_data_t::real_float, a_s, 2, ipiv_s, dpct::library_data_t::real_float, b_s, 2, info);
  //CHECK-NEXT:status = dpct::lapack::getrs(*handle, oneapi::mkl::transpose::nontrans, 2, 3, dpct::library_data_t::real_double, a_d, 2, ipiv_d, dpct::library_data_t::real_double, b_d, 2, info);
  //CHECK-NEXT:status = dpct::lapack::getrs(*handle, oneapi::mkl::transpose::nontrans, 2, 3, dpct::library_data_t::complex_float, a_c, 2, ipiv_c, dpct::library_data_t::complex_float, b_c, 2, info);
  //CHECK-NEXT:status = dpct::lapack::getrs(*handle, oneapi::mkl::transpose::nontrans, 2, 3, dpct::library_data_t::complex_double, a_z, 2, ipiv_z, dpct::library_data_t::complex_double, b_z, 2, info);
  status = cusolverDnXgetrs(handle, params, CUBLAS_OP_N, 2, 3, CUDA_R_32F, a_s, 2, ipiv_s, CUDA_R_32F, b_s, 2, info);
  status = cusolverDnXgetrs(handle, params, CUBLAS_OP_N, 2, 3, CUDA_R_64F, a_d, 2, ipiv_d, CUDA_R_64F, b_d, 2, info);
  status = cusolverDnXgetrs(handle, params, CUBLAS_OP_N, 2, 3, CUDA_C_32F, a_c, 2, ipiv_c, CUDA_C_32F, b_c, 2, info);
  status = cusolverDnXgetrs(handle, params, CUBLAS_OP_N, 2, 3, CUDA_C_64F, a_z, 2, ipiv_z, CUDA_C_64F, b_z, 2, info);

  //CHECK:/*
  //CHECK-NEXT:DPCT1027:{{[0-9]+}}: The call to cusolverDnDestroyParams was replaced with 0 because this call is redundant in SYCL.
  //CHECK-NEXT:*/
  //CHECK-NEXT:status = 0;
  //CHECK-NEXT:status = CHECK_SYCL_ERROR(handle = nullptr);
  status = cusolverDnDestroyParams(params);
  status = cusolverDnDestroy(handle);
}

void foo6() {
  cusolverStatus_t status;

  float* a_s;
  double* a_d;
  float2* a_c;
  double2* a_z;
  int64_t* ipiv_s;
  int64_t* ipiv_d;
  int64_t* ipiv_c;
  int64_t* ipiv_z;
  float* b_s;
  double* b_d;
  float2* b_c;
  double2* b_z;

  //CHECK:sycl::queue* handle;
  //CHECK-NEXT:status = CHECK_SYCL_ERROR(handle = &dpct::get_default_queue());
  cusolverDnHandle_t handle;
  status = cusolverDnCreate(&handle);

  //CHECK:int params;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1027:{{[0-9]+}}: The call to cusolverDnCreateParams was replaced with 0 because this call is redundant in SYCL.
  //CHECK-NEXT:*/
  //CHECK-NEXT:status = 0;
  cusolverDnParams_t params;
  status = cusolverDnCreateParams(&params);

  int *info;

  //CHECK:status = dpct::lapack::getrs(*handle, oneapi::mkl::transpose::nontrans, 2, 3, dpct::library_data_t::real_float, a_s, 2, ipiv_s, dpct::library_data_t::real_float, b_s, 2, info);
  //CHECK-NEXT:status = dpct::lapack::getrs(*handle, oneapi::mkl::transpose::nontrans, 2, 3, dpct::library_data_t::real_double, a_d, 2, ipiv_d, dpct::library_data_t::real_double, b_d, 2, info);
  //CHECK-NEXT:status = dpct::lapack::getrs(*handle, oneapi::mkl::transpose::nontrans, 2, 3, dpct::library_data_t::complex_float, a_c, 2, ipiv_c, dpct::library_data_t::complex_float, b_c, 2, info);
  //CHECK-NEXT:status = dpct::lapack::getrs(*handle, oneapi::mkl::transpose::nontrans, 2, 3, dpct::library_data_t::complex_double, a_z, 2, ipiv_z, dpct::library_data_t::complex_double, b_z, 2, info);
  status = cusolverDnGetrs(handle, params, CUBLAS_OP_N, 2, 3, CUDA_R_32F, a_s, 2, ipiv_s, CUDA_R_32F, b_s, 2, info);
  status = cusolverDnGetrs(handle, params, CUBLAS_OP_N, 2, 3, CUDA_R_64F, a_d, 2, ipiv_d, CUDA_R_64F, b_d, 2, info);
  status = cusolverDnGetrs(handle, params, CUBLAS_OP_N, 2, 3, CUDA_C_32F, a_c, 2, ipiv_c, CUDA_C_32F, b_c, 2, info);
  status = cusolverDnGetrs(handle, params, CUBLAS_OP_N, 2, 3, CUDA_C_64F, a_z, 2, ipiv_z, CUDA_C_64F, b_z, 2, info);

  //CHECK:/*
  //CHECK-NEXT:DPCT1027:{{[0-9]+}}: The call to cusolverDnDestroyParams was replaced with 0 because this call is redundant in SYCL.
  //CHECK-NEXT:*/
  //CHECK-NEXT:status = 0;
  //CHECK-NEXT:status = CHECK_SYCL_ERROR(handle = nullptr);
  status = cusolverDnDestroyParams(params);
  status = cusolverDnDestroy(handle);
}

void foo7() {
  void *a_s, *a_d, *a_c, *a_z;
  void *s_s, *s_d, *s_c, *s_z;
  void *u_s, *u_d, *u_c, *u_z;
  void *vt_s, *vt_d, *vt_c, *vt_z;
  cusolverDnHandle_t handle;
  size_t device_ws_size_s;
  size_t device_ws_size_d;
  size_t device_ws_size_c;
  size_t device_ws_size_z;
  size_t host_ws_size_s;
  size_t host_ws_size_d;
  size_t host_ws_size_c;
  size_t host_ws_size_z;
  cusolverDnParams_t params;

  //CHECK:dpct::lapack::gesvd_scratchpad_size(*handle, 'A', 'A', 2, 2, dpct::library_data_t::real_float, 2, dpct::library_data_t::real_float, 2, dpct::library_data_t::real_float, 2, &device_ws_size_s, &host_ws_size_s);
  //CHECK-NEXT:dpct::lapack::gesvd_scratchpad_size(*handle, 'A', 'A', 2, 2, dpct::library_data_t::real_double, 2, dpct::library_data_t::real_double, 2, dpct::library_data_t::real_double, 2, &device_ws_size_d, &host_ws_size_d);
  //CHECK-NEXT:dpct::lapack::gesvd_scratchpad_size(*handle, 'A', 'A', 2, 2, dpct::library_data_t::complex_float, 2, dpct::library_data_t::complex_float, 2, dpct::library_data_t::complex_float, 2, &device_ws_size_c, &host_ws_size_c);
  //CHECK-NEXT:dpct::lapack::gesvd_scratchpad_size(*handle, 'A', 'A', 2, 2, dpct::library_data_t::complex_double, 2, dpct::library_data_t::complex_double, 2, dpct::library_data_t::complex_double, 2, &device_ws_size_z, &host_ws_size_z);
  cusolverDnXgesvd_bufferSize(handle, params, 'A', 'A', 2, 2, CUDA_R_32F, a_s, 2, CUDA_R_32F, s_s, CUDA_R_32F, u_s, 2, CUDA_R_32F, vt_s, 2, CUDA_R_32F, &device_ws_size_s, &host_ws_size_s);
  cusolverDnXgesvd_bufferSize(handle, params, 'A', 'A', 2, 2, CUDA_R_64F, a_d, 2, CUDA_R_64F, s_d, CUDA_R_64F, u_d, 2, CUDA_R_64F, vt_d, 2, CUDA_R_64F, &device_ws_size_d, &host_ws_size_d);
  cusolverDnXgesvd_bufferSize(handle, params, 'A', 'A', 2, 2, CUDA_C_32F, a_c, 2, CUDA_R_32F, s_c, CUDA_C_32F, u_c, 2, CUDA_C_32F, vt_c, 2, CUDA_C_32F, &device_ws_size_c, &host_ws_size_c);
  cusolverDnXgesvd_bufferSize(handle, params, 'A', 'A', 2, 2, CUDA_C_64F, a_z, 2, CUDA_R_64F, s_z, CUDA_C_64F, u_z, 2, CUDA_C_64F, vt_z, 2, CUDA_C_64F, &device_ws_size_z, &host_ws_size_z);

  void* device_ws_s;
  void* device_ws_d;
  void* device_ws_c;
  void* device_ws_z;
  void* host_ws_s;
  void* host_ws_d;
  void* host_ws_c;
  void* host_ws_z;
  int *info;

  //CHECK:dpct::lapack::gesvd(*handle, 'A', 'A', 2, 2, dpct::library_data_t::real_float, a_s, 2, dpct::library_data_t::real_float, s_s, dpct::library_data_t::real_float, u_s, 2, dpct::library_data_t::real_float, vt_s, 2, device_ws_s, device_ws_size_s, info);
  //CHECK-NEXT:dpct::lapack::gesvd(*handle, 'A', 'A', 2, 2, dpct::library_data_t::real_double, a_d, 2, dpct::library_data_t::real_double, s_d, dpct::library_data_t::real_double, u_d, 2, dpct::library_data_t::real_double, vt_d, 2, device_ws_d, device_ws_size_d, info);
  //CHECK-NEXT:dpct::lapack::gesvd(*handle, 'A', 'A', 2, 2, dpct::library_data_t::complex_float, a_c, 2, dpct::library_data_t::real_float, s_c, dpct::library_data_t::complex_float, u_c, 2, dpct::library_data_t::complex_float, vt_c, 2, device_ws_c, device_ws_size_c, info);
  //CHECK-NEXT:dpct::lapack::gesvd(*handle, 'A', 'A', 2, 2, dpct::library_data_t::complex_double, a_z, 2, dpct::library_data_t::real_double, s_z, dpct::library_data_t::complex_double, u_z, 2, dpct::library_data_t::complex_double, vt_z, 2, device_ws_z, device_ws_size_z, info);
  cusolverDnXgesvd(handle, params, 'A', 'A', 2, 2, CUDA_R_32F, a_s, 2, CUDA_R_32F, s_s, CUDA_R_32F, u_s, 2, CUDA_R_32F, vt_s, 2, CUDA_R_32F, device_ws_s, device_ws_size_s, host_ws_s, host_ws_size_s, info);
  cusolverDnXgesvd(handle, params, 'A', 'A', 2, 2, CUDA_R_64F, a_d, 2, CUDA_R_64F, s_d, CUDA_R_64F, u_d, 2, CUDA_R_64F, vt_d, 2, CUDA_R_64F, device_ws_d, device_ws_size_d, host_ws_d, host_ws_size_d, info);
  cusolverDnXgesvd(handle, params, 'A', 'A', 2, 2, CUDA_C_32F, a_c, 2, CUDA_R_32F, s_c, CUDA_C_32F, u_c, 2, CUDA_C_32F, vt_c, 2, CUDA_C_32F, device_ws_c, device_ws_size_c, host_ws_c, host_ws_size_c, info);
  cusolverDnXgesvd(handle, params, 'A', 'A', 2, 2, CUDA_C_64F, a_z, 2, CUDA_R_64F, s_z, CUDA_C_64F, u_z, 2, CUDA_C_64F, vt_z, 2, CUDA_C_64F, device_ws_z, device_ws_size_z, host_ws_z, host_ws_size_z, info);
}

void foo8() {
  void *a_s, *a_d, *a_c, *a_z;
  void *s_s, *s_d, *s_c, *s_z;
  void *u_s, *u_d, *u_c, *u_z;
  void *vt_s, *vt_d, *vt_c, *vt_z;
  size_t device_ws_size_s;
  size_t device_ws_size_d;
  size_t device_ws_size_c;
  size_t device_ws_size_z;
  cusolverDnHandle_t handle;
  cusolverDnParams_t params;

  //CHECK:dpct::lapack::gesvd_scratchpad_size(*handle, 'A', 'A', 2, 2, dpct::library_data_t::real_float, 2, dpct::library_data_t::real_float, 2, dpct::library_data_t::real_float, 2, &device_ws_size_s);
  //CHECK-NEXT:dpct::lapack::gesvd_scratchpad_size(*handle, 'A', 'A', 2, 2, dpct::library_data_t::real_double, 2, dpct::library_data_t::real_double, 2, dpct::library_data_t::real_double, 2, &device_ws_size_d);
  //CHECK-NEXT:dpct::lapack::gesvd_scratchpad_size(*handle, 'A', 'A', 2, 2, dpct::library_data_t::complex_float, 2, dpct::library_data_t::complex_float, 2, dpct::library_data_t::complex_float, 2, &device_ws_size_c);
  //CHECK-NEXT:dpct::lapack::gesvd_scratchpad_size(*handle, 'A', 'A', 2, 2, dpct::library_data_t::complex_double, 2, dpct::library_data_t::complex_double, 2, dpct::library_data_t::complex_double, 2, &device_ws_size_z);
  cusolverDnGesvd_bufferSize(handle, params, 'A', 'A', 2, 2, CUDA_R_32F, a_s, 2, CUDA_R_32F, s_s, CUDA_R_32F, u_s, 2, CUDA_R_32F, vt_s, 2, CUDA_R_32F, &device_ws_size_s);
  cusolverDnGesvd_bufferSize(handle, params, 'A', 'A', 2, 2, CUDA_R_64F, a_d, 2, CUDA_R_64F, s_d, CUDA_R_64F, u_d, 2, CUDA_R_64F, vt_d, 2, CUDA_R_64F, &device_ws_size_d);
  cusolverDnGesvd_bufferSize(handle, params, 'A', 'A', 2, 2, CUDA_C_32F, a_c, 2, CUDA_R_32F, s_c, CUDA_C_32F, u_c, 2, CUDA_C_32F, vt_c, 2, CUDA_C_32F, &device_ws_size_c);
  cusolverDnGesvd_bufferSize(handle, params, 'A', 'A', 2, 2, CUDA_C_64F, a_z, 2, CUDA_R_64F, s_z, CUDA_C_64F, u_z, 2, CUDA_C_64F, vt_z, 2, CUDA_C_64F, &device_ws_size_z);

  void* device_ws_s;
  void* device_ws_d;
  void* device_ws_c;
  void* device_ws_z;
  int *info;

  //CHECK:dpct::lapack::gesvd(*handle, 'A', 'A', 2, 2, dpct::library_data_t::real_float, a_s, 2, dpct::library_data_t::real_float, s_s, dpct::library_data_t::real_float, u_s, 2, dpct::library_data_t::real_float, vt_s, 2, device_ws_s, device_ws_size_s, info);
  //CHECK-NEXT:dpct::lapack::gesvd(*handle, 'A', 'A', 2, 2, dpct::library_data_t::real_double, a_d, 2, dpct::library_data_t::real_double, s_d, dpct::library_data_t::real_double, u_d, 2, dpct::library_data_t::real_double, vt_d, 2, device_ws_d, device_ws_size_d, info);
  //CHECK-NEXT:dpct::lapack::gesvd(*handle, 'A', 'A', 2, 2, dpct::library_data_t::complex_float, a_c, 2, dpct::library_data_t::real_float, s_c, dpct::library_data_t::complex_float, u_c, 2, dpct::library_data_t::complex_float, vt_c, 2, device_ws_c, device_ws_size_c, info);
  //CHECK-NEXT:dpct::lapack::gesvd(*handle, 'A', 'A', 2, 2, dpct::library_data_t::complex_double, a_z, 2, dpct::library_data_t::real_double, s_z, dpct::library_data_t::complex_double, u_z, 2, dpct::library_data_t::complex_double, vt_z, 2, device_ws_z, device_ws_size_z, info);
  cusolverDnGesvd(handle, params, 'A', 'A', 2, 2, CUDA_R_32F, a_s, 2, CUDA_R_32F, s_s, CUDA_R_32F, u_s, 2, CUDA_R_32F, vt_s, 2, CUDA_R_32F, device_ws_s, device_ws_size_s, info);
  cusolverDnGesvd(handle, params, 'A', 'A', 2, 2, CUDA_R_64F, a_d, 2, CUDA_R_64F, s_d, CUDA_R_64F, u_d, 2, CUDA_R_64F, vt_d, 2, CUDA_R_64F, device_ws_d, device_ws_size_d, info);
  cusolverDnGesvd(handle, params, 'A', 'A', 2, 2, CUDA_C_32F, a_c, 2, CUDA_R_32F, s_c, CUDA_C_32F, u_c, 2, CUDA_C_32F, vt_c, 2, CUDA_C_32F, device_ws_c, device_ws_size_c, info);
  cusolverDnGesvd(handle, params, 'A', 'A', 2, 2, CUDA_C_64F, a_z, 2, CUDA_R_64F, s_z, CUDA_C_64F, u_z, 2, CUDA_C_64F, vt_z, 2, CUDA_C_64F, device_ws_z, device_ws_size_z, info);
}

void foo9() {
  void *a_s, *a_d, *a_c, *a_z;
  cusolverDnHandle_t handle;
  cusolverDnCreate(&handle);
  size_t device_ws_size_s;
  size_t device_ws_size_d;
  size_t device_ws_size_c;
  size_t device_ws_size_z;
  size_t host_ws_size_s;
  size_t host_ws_size_d;
  size_t host_ws_size_c;
  size_t host_ws_size_z;
  cusolverDnParams_t params;

  //CHECK:dpct::lapack::potrf_scratchpad_size(*handle, oneapi::mkl::uplo::lower, 3, dpct::library_data_t::real_float, 3, &device_ws_size_s, &host_ws_size_s);
  //CHECK-NEXT:dpct::lapack::potrf_scratchpad_size(*handle, oneapi::mkl::uplo::lower, 3, dpct::library_data_t::real_double, 3, &device_ws_size_d, &host_ws_size_d);
  //CHECK-NEXT:dpct::lapack::potrf_scratchpad_size(*handle, oneapi::mkl::uplo::lower, 3, dpct::library_data_t::complex_float, 3, &device_ws_size_c, &host_ws_size_c);
  //CHECK-NEXT:dpct::lapack::potrf_scratchpad_size(*handle, oneapi::mkl::uplo::lower, 3, dpct::library_data_t::complex_double, 3, &device_ws_size_z, &host_ws_size_z);
  cusolverDnXpotrf_bufferSize(handle, params, CUBLAS_FILL_MODE_LOWER, 3, CUDA_R_32F, a_s, 3, CUDA_R_32F, &device_ws_size_s, &host_ws_size_s);
  cusolverDnXpotrf_bufferSize(handle, params, CUBLAS_FILL_MODE_LOWER, 3, CUDA_R_64F, a_d, 3, CUDA_R_64F, &device_ws_size_d, &host_ws_size_d);
  cusolverDnXpotrf_bufferSize(handle, params, CUBLAS_FILL_MODE_LOWER, 3, CUDA_C_32F, a_c, 3, CUDA_R_32F, &device_ws_size_c, &host_ws_size_c);
  cusolverDnXpotrf_bufferSize(handle, params, CUBLAS_FILL_MODE_LOWER, 3, CUDA_C_64F, a_z, 3, CUDA_R_64F, &device_ws_size_z, &host_ws_size_z);

  void* device_ws_s;
  void* device_ws_d;
  void* device_ws_c;
  void* device_ws_z;
  void* host_ws_s;
  void* host_ws_d;
  void* host_ws_c;
  void* host_ws_z;
  int *info;

  //CHECK:dpct::lapack::potrf(*handle, oneapi::mkl::uplo::lower, 3, dpct::library_data_t::real_float, a_s, 3, device_ws_s, device_ws_size_s, info);
  //CHECK-NEXT:dpct::lapack::potrf(*handle, oneapi::mkl::uplo::lower, 3, dpct::library_data_t::real_double, a_d, 3, device_ws_d, device_ws_size_d, info);
  //CHECK-NEXT:dpct::lapack::potrf(*handle, oneapi::mkl::uplo::lower, 3, dpct::library_data_t::complex_float, a_c, 3, device_ws_c, device_ws_size_c, info);
  //CHECK-NEXT:dpct::lapack::potrf(*handle, oneapi::mkl::uplo::lower, 3, dpct::library_data_t::complex_double, a_z, 3, device_ws_z, device_ws_size_z, info);
  cusolverDnXpotrf(handle, params, CUBLAS_FILL_MODE_LOWER, 3, CUDA_R_32F, a_s, 3, CUDA_R_32F, device_ws_s, device_ws_size_s, host_ws_s, host_ws_size_s, info);
  cusolverDnXpotrf(handle, params, CUBLAS_FILL_MODE_LOWER, 3, CUDA_R_64F, a_d, 3, CUDA_R_64F, device_ws_d, device_ws_size_d, host_ws_d, host_ws_size_d, info);
  cusolverDnXpotrf(handle, params, CUBLAS_FILL_MODE_LOWER, 3, CUDA_C_32F, a_c, 3, CUDA_C_32F, device_ws_c, device_ws_size_c, host_ws_c, host_ws_size_c, info);
  cusolverDnXpotrf(handle, params, CUBLAS_FILL_MODE_LOWER, 3, CUDA_C_64F, a_z, 3, CUDA_C_64F, device_ws_z, device_ws_size_z, host_ws_z, host_ws_size_z, info);
}

void foo10() {
  void *a_s, *a_d, *a_c, *a_z;
  cusolverDnHandle_t handle;
  size_t device_ws_size_s;
  size_t device_ws_size_d;
  size_t device_ws_size_c;
  size_t device_ws_size_z;
  cusolverDnParams_t params;

  //CHECK:dpct::lapack::potrf_scratchpad_size(*handle, oneapi::mkl::uplo::lower, 3, dpct::library_data_t::real_float, 3, &device_ws_size_s);
  //CHECK-NEXT:dpct::lapack::potrf_scratchpad_size(*handle, oneapi::mkl::uplo::lower, 3, dpct::library_data_t::real_double, 3, &device_ws_size_d);
  //CHECK-NEXT:dpct::lapack::potrf_scratchpad_size(*handle, oneapi::mkl::uplo::lower, 3, dpct::library_data_t::complex_float, 3, &device_ws_size_c);
  //CHECK-NEXT:dpct::lapack::potrf_scratchpad_size(*handle, oneapi::mkl::uplo::lower, 3, dpct::library_data_t::complex_double, 3, &device_ws_size_z);
  cusolverDnPotrf_bufferSize(handle, params, CUBLAS_FILL_MODE_LOWER, 3, CUDA_R_32F, a_s, 3, CUDA_R_32F, &device_ws_size_s);
  cusolverDnPotrf_bufferSize(handle, params, CUBLAS_FILL_MODE_LOWER, 3, CUDA_R_64F, a_d, 3, CUDA_R_64F, &device_ws_size_d);
  cusolverDnPotrf_bufferSize(handle, params, CUBLAS_FILL_MODE_LOWER, 3, CUDA_C_32F, a_c, 3, CUDA_R_32F, &device_ws_size_c);
  cusolverDnPotrf_bufferSize(handle, params, CUBLAS_FILL_MODE_LOWER, 3, CUDA_C_64F, a_z, 3, CUDA_R_64F, &device_ws_size_z);

  void* device_ws_s;
  void* device_ws_d;
  void* device_ws_c;
  void* device_ws_z;
  int *info;

  //CHECK:dpct::lapack::potrf(*handle, oneapi::mkl::uplo::lower, 3, dpct::library_data_t::real_float, a_s, 3, device_ws_s, device_ws_size_s, info);
  //CHECK-NEXT:dpct::lapack::potrf(*handle, oneapi::mkl::uplo::lower, 3, dpct::library_data_t::real_double, a_d, 3, device_ws_d, device_ws_size_d, info);
  //CHECK-NEXT:dpct::lapack::potrf(*handle, oneapi::mkl::uplo::lower, 3, dpct::library_data_t::complex_float, a_c, 3, device_ws_c, device_ws_size_c, info);
  //CHECK-NEXT:dpct::lapack::potrf(*handle, oneapi::mkl::uplo::lower, 3, dpct::library_data_t::complex_double, a_z, 3, device_ws_z, device_ws_size_z, info);
  cusolverDnPotrf(handle, params, CUBLAS_FILL_MODE_LOWER, 3, CUDA_R_32F, a_s, 3, CUDA_R_32F, device_ws_s, device_ws_size_s, info);
  cusolverDnPotrf(handle, params, CUBLAS_FILL_MODE_LOWER, 3, CUDA_R_64F, a_d, 3, CUDA_R_64F, device_ws_d, device_ws_size_d, info);
  cusolverDnPotrf(handle, params, CUBLAS_FILL_MODE_LOWER, 3, CUDA_C_32F, a_c, 3, CUDA_C_32F, device_ws_c, device_ws_size_c, info);
  cusolverDnPotrf(handle, params, CUBLAS_FILL_MODE_LOWER, 3, CUDA_C_64F, a_z, 3, CUDA_C_64F, device_ws_z, device_ws_size_z, info);
}

void foo11() {
  void *a_s, *a_d, *a_c, *a_z;
  void *b_s, *b_d, *b_c, *b_z;
  cusolverDnHandle_t handle;
  cusolverDnParams_t params;
  int *info;

  //CHECK:dpct::lapack::potrs(*handle, oneapi::mkl::uplo::lower, 3, 1, dpct::library_data_t::real_float, a_s, 3, dpct::library_data_t::real_float, b_s, 3, info);
  //CHECK-NEXT:dpct::lapack::potrs(*handle, oneapi::mkl::uplo::lower, 3, 1, dpct::library_data_t::real_double, a_d, 3, dpct::library_data_t::real_double, b_d, 3, info);
  //CHECK-NEXT:dpct::lapack::potrs(*handle, oneapi::mkl::uplo::lower, 3, 1, dpct::library_data_t::complex_float, a_c, 3, dpct::library_data_t::complex_float, b_c, 3, info);
  //CHECK-NEXT:dpct::lapack::potrs(*handle, oneapi::mkl::uplo::lower, 3, 1, dpct::library_data_t::complex_double, a_z, 3, dpct::library_data_t::complex_double, b_z, 3, info);
  cusolverDnXpotrs(handle, params, CUBLAS_FILL_MODE_LOWER, 3, 1, CUDA_R_32F, a_s, 3, CUDA_R_32F, b_s, 3, info);
  cusolverDnXpotrs(handle, params, CUBLAS_FILL_MODE_LOWER, 3, 1, CUDA_R_64F, a_d, 3, CUDA_R_64F, b_d, 3, info);
  cusolverDnXpotrs(handle, params, CUBLAS_FILL_MODE_LOWER, 3, 1, CUDA_C_32F, a_c, 3, CUDA_C_32F, b_c, 3, info);
  cusolverDnXpotrs(handle, params, CUBLAS_FILL_MODE_LOWER, 3, 1, CUDA_C_64F, a_z, 3, CUDA_C_64F, b_z, 3, info);
}

void foo12() {
  void *a_s, *a_d, *a_c, *a_z;
  void *b_s, *b_d, *b_c, *b_z;
  cusolverDnHandle_t handle;
  cusolverDnParams_t params;
  int *info;

  //CHECK:dpct::lapack::potrs(*handle, oneapi::mkl::uplo::lower, 3, 1, dpct::library_data_t::real_float, a_s, 3, dpct::library_data_t::real_float, b_s, 3, info);
  //CHECK-NEXT:dpct::lapack::potrs(*handle, oneapi::mkl::uplo::lower, 3, 1, dpct::library_data_t::real_double, a_d, 3, dpct::library_data_t::real_double, b_d, 3, info);
  //CHECK-NEXT:dpct::lapack::potrs(*handle, oneapi::mkl::uplo::lower, 3, 1, dpct::library_data_t::complex_float, a_c, 3, dpct::library_data_t::complex_float, b_c, 3, info);
  //CHECK-NEXT:dpct::lapack::potrs(*handle, oneapi::mkl::uplo::lower, 3, 1, dpct::library_data_t::complex_double, a_z, 3, dpct::library_data_t::complex_double, b_z, 3, info);
  cusolverDnPotrs(handle, params, CUBLAS_FILL_MODE_LOWER, 3, 1, CUDA_R_32F, a_s, 3, CUDA_R_32F, b_s, 3, info);
  cusolverDnPotrs(handle, params, CUBLAS_FILL_MODE_LOWER, 3, 1, CUDA_R_64F, a_d, 3, CUDA_R_64F, b_d, 3, info);
  cusolverDnPotrs(handle, params, CUBLAS_FILL_MODE_LOWER, 3, 1, CUDA_C_32F, a_c, 3, CUDA_C_32F, b_c, 3, info);
  cusolverDnPotrs(handle, params, CUBLAS_FILL_MODE_LOWER, 3, 1, CUDA_C_64F, a_z, 3, CUDA_C_64F, b_z, 3, info);
}

void foo13() {
  float* a_s;
  double* a_d;
  float2* a_c;
  double2* a_z;
  float* w_s;
  double* w_d;
  float* w_c;
  double* w_z;

  cusolverDnHandle_t handle;
  cusolverDnParams_t params;

  size_t lwork_s;
  size_t lwork_d;
  size_t lwork_c;
  size_t lwork_z;

  int64_t h_meig_s;
  int64_t h_meig_d;
  int64_t h_meig_c;
  int64_t h_meig_z;
  float vlvu_s = 0;
  double vlvu_d = 0;
  float vlvu_c = 0;
  double vlvu_z = 0;

  //CHECK:dpct::lapack::syheevx_scratchpad_size(*handle, oneapi::mkl::job::vec, oneapi::mkl::rangev::all, oneapi::mkl::uplo::upper, 2, dpct::library_data_t::real_float, 2, &vlvu_s, &vlvu_s, 0, 0, dpct::library_data_t::real_float, &lwork_s);
  //CHECK-NEXT:dpct::lapack::syheevx_scratchpad_size(*handle, oneapi::mkl::job::vec, oneapi::mkl::rangev::all, oneapi::mkl::uplo::upper, 2, dpct::library_data_t::real_double, 2, &vlvu_d, &vlvu_d, 0, 0, dpct::library_data_t::real_double, &lwork_d);
  //CHECK-NEXT:dpct::lapack::syheevx_scratchpad_size(*handle, oneapi::mkl::job::vec, oneapi::mkl::rangev::all, oneapi::mkl::uplo::upper, 2, dpct::library_data_t::complex_float, 2, &vlvu_c, &vlvu_c, 0, 0, dpct::library_data_t::real_float, &lwork_c);
  //CHECK-NEXT:dpct::lapack::syheevx_scratchpad_size(*handle, oneapi::mkl::job::vec, oneapi::mkl::rangev::all, oneapi::mkl::uplo::upper, 2, dpct::library_data_t::complex_double, 2, &vlvu_z, &vlvu_z, 0, 0, dpct::library_data_t::real_double, &lwork_z);
  cusolverDnSyevdx_bufferSize(handle, params, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, CUDA_R_32F, a_s, 2, &vlvu_s, &vlvu_s, 0, 0, &h_meig_s, CUDA_R_32F, w_s, CUDA_R_32F, &lwork_s);
  cusolverDnSyevdx_bufferSize(handle, params, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, CUDA_R_64F, a_d, 2, &vlvu_d, &vlvu_d, 0, 0, &h_meig_d, CUDA_R_64F, w_d, CUDA_R_64F, &lwork_d);
  cusolverDnSyevdx_bufferSize(handle, params, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, CUDA_C_32F, a_c, 2, &vlvu_c, &vlvu_c, 0, 0, &h_meig_c, CUDA_R_32F, w_c, CUDA_C_32F, &lwork_c);
  cusolverDnSyevdx_bufferSize(handle, params, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, CUDA_C_64F, a_z, 2, &vlvu_z, &vlvu_z, 0, 0, &h_meig_z, CUDA_R_64F, w_z, CUDA_C_64F, &lwork_z);

  void* device_ws_s;
  void* device_ws_d;
  void* device_ws_c;
  void* device_ws_z;

  int *info;

  //CHECK:dpct::lapack::syheevx(*handle, oneapi::mkl::job::vec, oneapi::mkl::rangev::all, oneapi::mkl::uplo::upper, 2, dpct::library_data_t::real_float, a_s, 2, &vlvu_s, &vlvu_s, 0, 0, &h_meig_s, dpct::library_data_t::real_float, w_s, device_ws_s, lwork_s, info);
  //CHECK-NEXT:dpct::lapack::syheevx(*handle, oneapi::mkl::job::vec, oneapi::mkl::rangev::all, oneapi::mkl::uplo::upper, 2, dpct::library_data_t::real_double, a_d, 2, &vlvu_d, &vlvu_d, 0, 0, &h_meig_d, dpct::library_data_t::real_double, w_d, device_ws_d, lwork_d, info);
  //CHECK-NEXT:dpct::lapack::syheevx(*handle, oneapi::mkl::job::vec, oneapi::mkl::rangev::all, oneapi::mkl::uplo::upper, 2, dpct::library_data_t::complex_float, a_c, 2, &vlvu_c, &vlvu_c, 0, 0, &h_meig_c, dpct::library_data_t::real_float, w_c, device_ws_c, lwork_c, info);
  //CHECK-NEXT:dpct::lapack::syheevx(*handle, oneapi::mkl::job::vec, oneapi::mkl::rangev::all, oneapi::mkl::uplo::upper, 2, dpct::library_data_t::complex_double, a_z, 2, &vlvu_z, &vlvu_z, 0, 0, &h_meig_z, dpct::library_data_t::real_double, w_z, device_ws_z, lwork_z, info);
  cusolverDnSyevdx(handle, params, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, CUDA_R_32F, a_s, 2, &vlvu_s, &vlvu_s, 0, 0, &h_meig_s, CUDA_R_32F, w_s, CUDA_R_32F, device_ws_s, lwork_s, info);
  cusolverDnSyevdx(handle, params, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, CUDA_R_64F, a_d, 2, &vlvu_d, &vlvu_d, 0, 0, &h_meig_d, CUDA_R_64F, w_d, CUDA_R_64F, device_ws_d, lwork_d, info);
  cusolverDnSyevdx(handle, params, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, CUDA_C_32F, a_c, 2, &vlvu_c, &vlvu_c, 0, 0, &h_meig_c, CUDA_R_32F, w_c, CUDA_C_32F, device_ws_c, lwork_c, info);
  cusolverDnSyevdx(handle, params, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, CUDA_C_64F, a_z, 2, &vlvu_z, &vlvu_z, 0, 0, &h_meig_z, CUDA_R_64F, w_z, CUDA_C_64F, device_ws_z, lwork_z, info);
}

void foo14() {
  float* a_s;
  double* a_d;
  float2* a_c;
  double2* a_z;
  float* w_s;
  double* w_d;
  float* w_c;
  double* w_z;

  cusolverDnHandle_t handle;
  cusolverDnParams_t params;

  size_t lwork_s;
  size_t lwork_d;
  size_t lwork_c;
  size_t lwork_z;
  size_t lwork_host_s;
  size_t lwork_host_d;
  size_t lwork_host_c;
  size_t lwork_host_z;

  int64_t h_meig_s;
  int64_t h_meig_d;
  int64_t h_meig_c;
  int64_t h_meig_z;
  float vlvu_s = 0;
  double vlvu_d = 0;
  float vlvu_c = 0;
  double vlvu_z = 0;

  //CHECK:dpct::lapack::syheevx_scratchpad_size(*handle, oneapi::mkl::job::vec, oneapi::mkl::rangev::all, oneapi::mkl::uplo::upper, 2, dpct::library_data_t::real_float, 2, &vlvu_s, &vlvu_s, 0, 0, dpct::library_data_t::real_float, &lwork_s, &lwork_host_s);
  //CHECK-NEXT:dpct::lapack::syheevx_scratchpad_size(*handle, oneapi::mkl::job::vec, oneapi::mkl::rangev::all, oneapi::mkl::uplo::upper, 2, dpct::library_data_t::real_double, 2, &vlvu_d, &vlvu_d, 0, 0, dpct::library_data_t::real_double, &lwork_d, &lwork_host_d);
  //CHECK-NEXT:dpct::lapack::syheevx_scratchpad_size(*handle, oneapi::mkl::job::vec, oneapi::mkl::rangev::all, oneapi::mkl::uplo::upper, 2, dpct::library_data_t::complex_float, 2, &vlvu_c, &vlvu_c, 0, 0, dpct::library_data_t::real_float, &lwork_c, &lwork_host_c);
  //CHECK-NEXT:dpct::lapack::syheevx_scratchpad_size(*handle, oneapi::mkl::job::vec, oneapi::mkl::rangev::all, oneapi::mkl::uplo::upper, 2, dpct::library_data_t::complex_double, 2, &vlvu_z, &vlvu_z, 0, 0, dpct::library_data_t::real_double, &lwork_z, &lwork_host_z);
  cusolverDnXsyevdx_bufferSize(handle, params, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, CUDA_R_32F, a_s, 2, &vlvu_s, &vlvu_s, 0, 0, &h_meig_s, CUDA_R_32F, w_s, CUDA_R_32F, &lwork_s, &lwork_host_s);
  cusolverDnXsyevdx_bufferSize(handle, params, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, CUDA_R_64F, a_d, 2, &vlvu_d, &vlvu_d, 0, 0, &h_meig_d, CUDA_R_64F, w_d, CUDA_R_64F, &lwork_d, &lwork_host_d);
  cusolverDnXsyevdx_bufferSize(handle, params, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, CUDA_C_32F, a_c, 2, &vlvu_c, &vlvu_c, 0, 0, &h_meig_c, CUDA_R_32F, w_c, CUDA_C_32F, &lwork_c, &lwork_host_c);
  cusolverDnXsyevdx_bufferSize(handle, params, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, CUDA_C_64F, a_z, 2, &vlvu_z, &vlvu_z, 0, 0, &h_meig_z, CUDA_R_64F, w_z, CUDA_C_64F, &lwork_z, &lwork_host_z);

  void* device_ws_s;
  void* device_ws_d;
  void* device_ws_c;
  void* device_ws_z;
  void* host_ws_s;
  void* host_ws_d;
  void* host_ws_c;
  void* host_ws_z;

  int *info;

  //CHECK:dpct::lapack::syheevx(*handle, oneapi::mkl::job::vec, oneapi::mkl::rangev::all, oneapi::mkl::uplo::upper, 2, dpct::library_data_t::real_float, a_s, 2, &vlvu_s, &vlvu_s, 0, 0, &h_meig_s, dpct::library_data_t::real_float, w_s, device_ws_s, lwork_s, info);
  //CHECK-NEXT:dpct::lapack::syheevx(*handle, oneapi::mkl::job::vec, oneapi::mkl::rangev::all, oneapi::mkl::uplo::upper, 2, dpct::library_data_t::real_double, a_d, 2, &vlvu_d, &vlvu_d, 0, 0, &h_meig_d, dpct::library_data_t::real_double, w_d, device_ws_d, lwork_d, info);
  //CHECK-NEXT:dpct::lapack::syheevx(*handle, oneapi::mkl::job::vec, oneapi::mkl::rangev::all, oneapi::mkl::uplo::upper, 2, dpct::library_data_t::complex_float, a_c, 2, &vlvu_c, &vlvu_c, 0, 0, &h_meig_c, dpct::library_data_t::real_float, w_c, device_ws_c, lwork_c, info);
  //CHECK-NEXT:dpct::lapack::syheevx(*handle, oneapi::mkl::job::vec, oneapi::mkl::rangev::all, oneapi::mkl::uplo::upper, 2, dpct::library_data_t::complex_double, a_z, 2, &vlvu_z, &vlvu_z, 0, 0, &h_meig_z, dpct::library_data_t::real_double, w_z, device_ws_z, lwork_z, info);
  cusolverDnXsyevdx(handle, params, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, CUDA_R_32F, a_s, 2, &vlvu_s, &vlvu_s, 0, 0, &h_meig_s, CUDA_R_32F, w_s, CUDA_R_32F, device_ws_s, lwork_s, host_ws_s, lwork_host_s, info);
  cusolverDnXsyevdx(handle, params, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, CUDA_R_64F, a_d, 2, &vlvu_d, &vlvu_d, 0, 0, &h_meig_d, CUDA_R_64F, w_d, CUDA_R_64F, device_ws_d, lwork_d, host_ws_d, lwork_host_d, info);
  cusolverDnXsyevdx(handle, params, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, CUDA_C_32F, a_c, 2, &vlvu_c, &vlvu_c, 0, 0, &h_meig_c, CUDA_R_32F, w_c, CUDA_C_32F, device_ws_c, lwork_c, host_ws_c, lwork_host_c, info);
  cusolverDnXsyevdx(handle, params, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_ALL, CUBLAS_FILL_MODE_UPPER, 2, CUDA_C_64F, a_z, 2, &vlvu_z, &vlvu_z, 0, 0, &h_meig_z, CUDA_R_64F, w_z, CUDA_C_64F, device_ws_z, lwork_z, host_ws_z, lwork_host_z, info);
}
