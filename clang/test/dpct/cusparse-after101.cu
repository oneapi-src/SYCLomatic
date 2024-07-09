// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0
// RUN: dpct --format-range=none --out-root %T/cusparse-after101 %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --input-file %T/cusparse-after101/cusparse-after101.dp.cpp --match-full-lines %s

#include <cusparse_v2.h>

void foo0() {
  // CHECK: dpct::sparse::conversion_scope a = dpct::sparse::conversion_scope::index;
  // CHECK-NEXT: dpct::sparse::conversion_scope b = dpct::sparse::conversion_scope::index_and_value;
  cusparseAction_t a = CUSPARSE_ACTION_SYMBOLIC;
  cusparseAction_t b = CUSPARSE_ACTION_NUMERIC;
}

void foo1() {
  cusparseHandle_t handle;
  float* a_val;
  int* a_row_ptr;
  int* a_col_ind;
  float* b_val;
  int* b_col_ptr;
  int* b_row_ind;

  size_t ws_size = 0;
  void *ws;

  // CHECK: ws_size = 0;
  // CHECK-NEXT: dpct::sparse::csr2csc(handle->get_queue(), 3, 4, 7, a_val, a_row_ptr, a_col_ind, b_val, b_col_ptr, b_row_ind, dpct::library_data_t::real_float, dpct::sparse::conversion_scope::index_and_value, oneapi::mkl::index_base::zero);
  cusparseCsr2cscEx2_bufferSize(handle, 3, 4, 7, a_val, a_row_ptr, a_col_ind, b_val, b_col_ptr, b_row_ind, CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, &ws_size);
  cusparseCsr2cscEx2(handle, 3, 4, 7, a_val, a_row_ptr, a_col_ind, b_val, b_col_ptr, b_row_ind, CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, ws);
}
