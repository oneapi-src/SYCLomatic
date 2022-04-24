// RUN: dpct --format-range=none -out-root %T/output %S/test1.cu %S/test2.cu %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/output/test3.dp.cpp --match-full-lines %s

#include "cublas_v2.h"

int foo () {
  cublasStatus_t s;
  cublasHandle_t handle;
  int N = 275;
  float *x1;
  int *result;

  //CHECK:int64_t* res_temp_ptr_ct5 = sycl::malloc_shared<int64_t>(1, dpct::get_default_queue());
  //CHECK-NEXT:oneapi::mkl::blas::column_major::iamax(*handle, N, x1, N, res_temp_ptr_ct5).wait();
  //CHECK-NEXT:int res_temp_host_ct6 = (int)*res_temp_ptr_ct5;
  //CHECK-NEXT:dpct::dpct_memcpy(result, &res_temp_host_ct6, sizeof(int));
  //CHECK-NEXT:sycl::free(res_temp_ptr_ct5, dpct::get_default_queue());
  cublasIsamax(handle, N, x1, N, result);
}
