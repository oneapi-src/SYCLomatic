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
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:status = (handle = &dpct::get_default_queue(), 0);
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
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:status = (handle = nullptr, 0);
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
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:status = (handle = &dpct::get_default_queue(), 0);
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
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:status = (handle = nullptr, 0);
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
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:status = (handle = &dpct::get_default_queue(), 0);
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
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:status = (handle = nullptr, 0);
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
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:status = (handle = &dpct::get_default_queue(), 0);
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
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:status = (handle = nullptr, 0);
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
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:status = (handle = &dpct::get_default_queue(), 0);
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
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:status = (handle = nullptr, 0);
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
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:status = (handle = &dpct::get_default_queue(), 0);
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
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:status = (handle = nullptr, 0);
  status = cusolverDnDestroyParams(params);
  status = cusolverDnDestroy(handle);
}
