// UNSUPPORTED: v12.0, v12.1, v12.2
// UNSUPPORTED: cuda-12.0, cuda-12.1, cuda-12.2
// RUN: dpct --format-range=none --usm-level=none --out-root %T/cusparse_before_12 %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --input-file %T/cusparse_before_12/cusparse_before_12.dp.cpp --match-full-lines %s
#include <cstdio>
#include <cusparse_v2.h>
#include <cuda_runtime.h>

void foo() {
  cusparseHandle_t handle;
  cusparseMatDescr_t descrA;
  cusparseSolveAnalysisInfo_t info;
  cusparseCreateSolveAnalysisInfo(&info);

  float *a_s_val;
  int *a_row_ptr;
  int *a_col_ind;
  float *f_s;
  float *x_s;
  float alpha_s = 1;

  //CHECK:dpct::sparse::optimize_csrsv(*handle, oneapi::mkl::transpose::nontrans, 3, descrA, a_s_val, dpct::library_data_t::real_float, a_row_ptr, a_col_ind, info);
  //CHECK-NEXT:dpct::sparse::csrsv(*handle, oneapi::mkl::transpose::nontrans, 3, &alpha_s, dpct::library_data_t::real_float, descrA, a_s_val, dpct::library_data_t::real_float, a_row_ptr, a_col_ind, info, f_s, dpct::library_data_t::real_float, x_s, dpct::library_data_t::real_float);
  cusparseCsrsv_analysisEx(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, 6, descrA, a_s_val, CUDA_R_32F, a_row_ptr, a_col_ind, info, CUDA_R_32F);
  cusparseCsrsv_solveEx(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 3, &alpha_s, CUDA_R_32F, descrA, a_s_val, CUDA_R_32F, a_row_ptr, a_col_ind, info, f_s, CUDA_R_32F, x_s, CUDA_R_32F, CUDA_R_32F);
}
