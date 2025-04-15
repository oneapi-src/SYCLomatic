// UNSUPPORTED: v8.0, v12.0, v12.1, v12.2, v12.3, v12.4, v12.5, v12.6, v12.8
// UNSUPPORTED: cuda-8.0, cuda-12.0, cuda-12.1, cuda-12.2, cuda-12.3, cuda-12.4, cuda-12.5, cuda-12.6, cuda-12.8
// RUN: dpct --format-range=none --usm-level=none --out-root %T/cusparse_9_to_11 %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/cusparse_9_to_11/cusparse_9_to_11.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/cusparse_9_to_11/cusparse_9_to_11.dp.cpp -o %T/cusparse_9_to_11/cusparse_9_to_11.dp.o %}

#include <cusparse_v2.h>

void foo() {
  cusparseHandle_t handle;
  int m;
  int nrhs;
  int nnz;
  cusparseMatDescr_t descrA;
  float *val_s;
  double *val_d;
  float2 *val_c;
  double2 *val_z;
  int *row_ptr;
  int *col_ind;
  float *b_s;
  double *b_d;
  float2 *b_c;
  double2 *b_z;
  float alpha_s = 1;
  double alpha_d = 1;
  float2 alpha_c = float2{1, 0};
  double2 alpha_z = double2{1, 0};
  size_t buffer_size;
  void* buffer;

  // CHECK: int policy;
  // CHECK-NEXT: std::shared_ptr<dpct::sparse::optimize_info> info;
  // CHECK-NEXT: info = std::make_shared<dpct::sparse::optimize_info>();
  cusparseSolvePolicy_t policy;
  csrsm2Info_t info;
  cusparseCreateCsrsm2Info(&info);

  // CHECK: buffer_size = 0;
  // CHECK-NEXT: buffer_size = 0;
  // CHECK-NEXT: buffer_size = 0;
  // CHECK-NEXT: buffer_size = 0;
  cusparseScsrsm2_bufferSizeExt(handle, 0, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, m, nrhs, nnz, &alpha_s, descrA, val_s, row_ptr, col_ind, b_s, nrhs, info, policy, &buffer_size);
  cusparseDcsrsm2_bufferSizeExt(handle, 0, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, m, nrhs, nnz, &alpha_d, descrA, val_d, row_ptr, col_ind, b_d, nrhs, info, policy, &buffer_size);
  cusparseCcsrsm2_bufferSizeExt(handle, 0, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, m, nrhs, nnz, &alpha_c, descrA, val_c, row_ptr, col_ind, b_c, nrhs, info, policy, &buffer_size);
  cusparseZcsrsm2_bufferSizeExt(handle, 0, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, m, nrhs, nnz, &alpha_z, descrA, val_z, row_ptr, col_ind, b_z, nrhs, info, policy, &buffer_size);

  // CHECK: dpct::sparse::optimize_csrsm(handle->get_queue(), oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans, m, nrhs, descrA, val_s, row_ptr, col_ind, info);
  // CHECK-NEXT: dpct::sparse::optimize_csrsm(handle->get_queue(), oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans, m, nrhs, descrA, val_d, row_ptr, col_ind, info);
  // CHECK-NEXT: dpct::sparse::optimize_csrsm(handle->get_queue(), oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans, m, nrhs, descrA, val_c, row_ptr, col_ind, info);
  // CHECK-NEXT: dpct::sparse::optimize_csrsm(handle->get_queue(), oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans, m, nrhs, descrA, val_z, row_ptr, col_ind, info);
  cusparseScsrsm2_analysis(handle, 0, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, m, nrhs, nnz, &alpha_s, descrA, val_s, row_ptr, col_ind, b_s, nrhs, info, policy, buffer);
  cusparseDcsrsm2_analysis(handle, 0, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, m, nrhs, nnz, &alpha_d, descrA, val_d, row_ptr, col_ind, b_d, nrhs, info, policy, buffer);
  cusparseCcsrsm2_analysis(handle, 0, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, m, nrhs, nnz, &alpha_c, descrA, val_c, row_ptr, col_ind, b_c, nrhs, info, policy, buffer);
  cusparseZcsrsm2_analysis(handle, 0, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, m, nrhs, nnz, &alpha_z, descrA, val_z, row_ptr, col_ind, b_z, nrhs, info, policy, buffer);

  // CHECK: dpct::sparse::csrsm(handle->get_queue(), oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans, m, nrhs, &alpha_s, descrA, val_s, row_ptr, col_ind, b_s, nrhs, info);
  // CHECK-NEXT: dpct::sparse::csrsm(handle->get_queue(), oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans, m, nrhs, &alpha_d, descrA, val_d, row_ptr, col_ind, b_d, nrhs, info);
  // CHECK-NEXT: dpct::sparse::csrsm(handle->get_queue(), oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans, m, nrhs, &alpha_c, descrA, val_c, row_ptr, col_ind, b_c, nrhs, info);
  // CHECK-NEXT: dpct::sparse::csrsm(handle->get_queue(), oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans, m, nrhs, &alpha_z, descrA, val_z, row_ptr, col_ind, b_z, nrhs, info);
  cusparseScsrsm2_solve(handle, 0, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, m, nrhs, nnz, &alpha_s, descrA, val_s, row_ptr, col_ind, b_s, nrhs, info, policy, buffer);
  cusparseDcsrsm2_solve(handle, 0, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, m, nrhs, nnz, &alpha_d, descrA, val_d, row_ptr, col_ind, b_d, nrhs, info, policy, buffer);
  cusparseCcsrsm2_solve(handle, 0, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, m, nrhs, nnz, &alpha_c, descrA, val_c, row_ptr, col_ind, b_c, nrhs, info, policy, buffer);
  cusparseZcsrsm2_solve(handle, 0, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, m, nrhs, nnz, &alpha_z, descrA, val_z, row_ptr, col_ind, b_z, nrhs, info, policy, buffer);

  // CHECK: info.reset();
  cusparseDestroyCsrsm2Info(info);
}
