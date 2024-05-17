// RUN: dpct --format-range=none --out-root %T/cublaslt %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --input-file %T/cublaslt/cublaslt.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/cublaslt/cublaslt.dp.cpp -o %T/cublaslt/cublaslt.dp.o %}

#include "cublasLt.h"

void foo1 () {
  cublasLtHandle_t ltHandle;
  cublasLtCreate(&ltHandle);
  cublasLtDestroy(ltHandle);

  cublasLtMatrixLayout_t matLayout;
  cudaDataType type;
  uint64_t rows;
  uint64_t cols;
  int64_t ld;
  cublasLtMatrixLayoutCreate(&matLayout, type, rows, cols, ld);

  cublasLtMatrixLayoutAttribute_t attr1;
  void *buf1;
  size_t sizeInBytes1;
  size_t *sizeWritten1;
  cublasLtMatrixLayoutGetAttribute(matLayout, attr1, buf1, sizeInBytes1, sizeWritten1);
  cublasLtMatrixLayoutSetAttribute(matLayout, attr1, buf1, sizeInBytes1);
  cublasLtMatrixLayoutDestroy(matLayout);

  cublasLtMatmulDesc_t matmulDesc;
  cublasComputeType_t computeType;
  cudaDataType_t scaleType;
  cublasLtMatmulDescCreate(&matmulDesc, computeType, scaleType);

  cublasLtMatmulDescAttributes_t attr2;
  void *buf2;
  size_t sizeInBytes2;
  size_t *sizeWritten2;
  cublasLtMatmulDescGetAttribute(matmulDesc, attr2, buf2, sizeInBytes2, sizeWritten2);
  cublasLtMatmulDescSetAttribute(matmulDesc, attr2, buf2, sizeInBytes2);
  cublasLtMatmulDescDestroy(matmulDesc);
}

void foo2() {
  cublasLtHandle_t lightHandle;
  cublasLtMatmulDesc_t computeDesc;
  const void *alpha;
  const void *A;
  cublasLtMatrixLayout_t Adesc;
  const void *B;
  cublasLtMatrixLayout_t Bdesc;
  const void *beta;
  const void *C;
  cublasLtMatrixLayout_t Cdesc;
  void *D;
  cublasLtMatrixLayout_t Ddesc;
  const cublasLtMatmulAlgo_t *algo;
  void *workspace;
  size_t workspaceSizeInBytes;
  cudaStream_t stream;
  cublasLtMatmul(lightHandle, computeDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, D, Ddesc, algo, workspace, workspaceSizeInBytes, stream);
}

void foo3() {
  cublasLtOrder_t a;
  a = CUBLASLT_ORDER_COL;
  a = CUBLASLT_ORDER_ROW;
  a = CUBLASLT_ORDER_COL32;
  a = CUBLASLT_ORDER_COL4_4R2_8C;
  a = CUBLASLT_ORDER_COL32_2R_4R4;
  cublasLtPointerMode_t b;
  b = CUBLASLT_POINTER_MODE_HOST;
  b = CUBLASLT_POINTER_MODE_DEVICE;
  b = CUBLASLT_POINTER_MODE_DEVICE_VECTOR;
  b = CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO;
  b = CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST;
  cublasLtMatrixLayoutAttribute_t c;
  c = CUBLASLT_MATRIX_LAYOUT_TYPE;
  c = CUBLASLT_MATRIX_LAYOUT_ORDER;
  c = CUBLASLT_MATRIX_LAYOUT_ROWS;
  c = CUBLASLT_MATRIX_LAYOUT_COLS;
  c = CUBLASLT_MATRIX_LAYOUT_LD;
  c = CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT;
  c = CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET;
  c = CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET;
  cublasLtMatmulDescAttributes_t d;
  d = CUBLASLT_MATMUL_DESC_COMPUTE_TYPE;
  d = CUBLASLT_MATMUL_DESC_SCALE_TYPE;
  d = CUBLASLT_MATMUL_DESC_POINTER_MODE;
  d = CUBLASLT_MATMUL_DESC_TRANSA;
  d = CUBLASLT_MATMUL_DESC_TRANSB;
  d = CUBLASLT_MATMUL_DESC_TRANSC;
  d = CUBLASLT_MATMUL_DESC_FILL_MODE;
  d = CUBLASLT_MATMUL_DESC_EPILOGUE;
  d = CUBLASLT_MATMUL_DESC_BIAS_POINTER;
  d = CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE;
  d = CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER;
  d = CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD;
  d = CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE;
  d = CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE;
  d = CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET;
  d = CUBLASLT_MATMUL_DESC_A_SCALE_POINTER;
  d = CUBLASLT_MATMUL_DESC_B_SCALE_POINTER;
  d = CUBLASLT_MATMUL_DESC_C_SCALE_POINTER;
  d = CUBLASLT_MATMUL_DESC_D_SCALE_POINTER;
  d = CUBLASLT_MATMUL_DESC_AMAX_D_POINTER;
  d = CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE;
  d = CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER;
  d = CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_AMAX_POINTER;
  d = CUBLASLT_MATMUL_DESC_FAST_ACCUM;
  d = CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE;
  d = CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_IN_COUNTERS_POINTER;
  d = CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_OUT_COUNTERS_POINTER;
  d = CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_ROWS;
  d = CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_COLS;
}
