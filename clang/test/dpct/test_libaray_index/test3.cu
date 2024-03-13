// RUN: dpct --format-range=none -out-root %T/output %S/test1.cu %S/test2.cu %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/output/test3.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/output/test3.dp.cpp -o %T/output/test3.dp.o %}

#include "cublas_v2.h"

int foo () {
  cublasStatus_t s;
  cublasHandle_t handle;
  int N = 275;
  float *x1;
  int *result;

  //CHECK:int64_t* res_temp_ptr_ct5 = sycl::malloc_shared<int64_t>(1, q_ct1);
  //CHECK-NEXT:oneapi::mkl::blas::column_major::iamax(handle->get_queue(), N, x1, N, res_temp_ptr_ct5, oneapi::mkl::index_base::one).wait();
  //CHECK-NEXT:int res_temp_host_ct6 = (int)*res_temp_ptr_ct5;
  //CHECK-NEXT:dpct::dpct_memcpy(result, &res_temp_host_ct6, sizeof(int));
  //CHECK-NEXT:sycl::free(res_temp_ptr_ct5, q_ct1);
  cublasIsamax(handle, N, x1, N, result);
}
