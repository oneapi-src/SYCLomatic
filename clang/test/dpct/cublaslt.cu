// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-11.8, cuda-12.0, cuda-12.1, cuda-12.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6, v11.7, v11.8, v12.0, v12.1, v12.2
// RUN: dpct --format-range=none --out-root %T/cublaslt %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --input-file %T/cublaslt/cublaslt.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/cublaslt/cublaslt.dp.cpp -o %T/cublaslt/cublaslt.dp.o %}

#include "cublasLt.h"

void foo1 () {
  // CHECK: dpct::blas_gemm::experimental::descriptor_ptr ltHandle;
  // CHECK-NEXT: ltHandle = new dpct::blas_gemm::experimental::descriptor();
  // CHECK-NEXT: delete (ltHandle);
  cublasLtHandle_t ltHandle;
  cublasLtCreate(&ltHandle);
  cublasLtDestroy(ltHandle);

  // CHECK: dpct::blas_gemm::experimental::matrix_layout_ptr matLayout;
  // CHECK-NEXT: dpct::library_data_t type;
  // CHECK-NEXT: uint64_t rows;
  // CHECK-NEXT: uint64_t cols;
  // CHECK-NEXT: int64_t ld;
  // CHECK-NEXT: matLayout = new dpct::blas_gemm::experimental::matrix_layout_t(type, rows, cols, ld);
  cublasLtMatrixLayout_t matLayout;
  cudaDataType type;
  uint64_t rows;
  uint64_t cols;
  int64_t ld;
  cublasLtMatrixLayoutCreate(&matLayout, type, rows, cols, ld);

  // CHECK: dpct::blas_gemm::experimental::matrix_layout_t::attribute attr1;
  // CHECK-NEXT: void *buf1;
  // CHECK-NEXT: size_t sizeInBytes1;
  // CHECK-NEXT: size_t *sizeWritten1;
  // CHECK-NEXT: matLayout->get_attribute(attr1, buf1);
  // CHECK-NEXT: matLayout->set_attribute(attr1, buf1);
  // CHECK-NEXT: delete (matLayout);
  cublasLtMatrixLayoutAttribute_t attr1;
  void *buf1;
  size_t sizeInBytes1;
  size_t *sizeWritten1;
  cublasLtMatrixLayoutGetAttribute(matLayout, attr1, buf1, sizeInBytes1, sizeWritten1);
  cublasLtMatrixLayoutSetAttribute(matLayout, attr1, buf1, sizeInBytes1);
  cublasLtMatrixLayoutDestroy(matLayout);

  // CHECK: dpct::blas_gemm::experimental::matmul_desc_ptr matmulDesc;
  // CHECK-NEXT: dpct::compute_type computeType;
  // CHECK-NEXT: dpct::library_data_t scaleType;
  // CHECK-NEXT: matmulDesc = new dpct::blas_gemm::experimental::matmul_desc_t(computeType, scaleType);
  cublasLtMatmulDesc_t matmulDesc;
  cublasComputeType_t computeType;
  cudaDataType_t scaleType;
  cublasLtMatmulDescCreate(&matmulDesc, computeType, scaleType);

  // CHECK: dpct::blas_gemm::experimental::matmul_desc_t::attribute attr2;
  // CHECK-NEXT: void *buf2;
  // CHECK-NEXT: size_t sizeInBytes2;
  // CHECK-NEXT: size_t *sizeWritten2;
  // CHECK-NEXT: matmulDesc->get_attribute(attr2, buf2);
  // CHECK-NEXT: matmulDesc->set_attribute(attr2, buf2);
  // CHECK-NEXT: delete (matmulDesc);
  cublasLtMatmulDescAttributes_t attr2;
  void *buf2;
  size_t sizeInBytes2;
  size_t *sizeWritten2;
  cublasLtMatmulDescGetAttribute(matmulDesc, attr2, buf2, sizeInBytes2, sizeWritten2);
  cublasLtMatmulDescSetAttribute(matmulDesc, attr2, buf2, sizeInBytes2);
  cublasLtMatmulDescDestroy(matmulDesc);

  // CHECK: int matmulPreference;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cublasLtMatmulPreferenceCreate was removed because this functionality is redundant in SYCL.
  // CHECK-NEXT: */
  // CHECK-NEXT: void *buf3;
  // CHECK-NEXT: size_t sizeInBytes3;
  // CHECK-NEXT: size_t *sizeWritten3;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cublasLtMatmulPreferenceGetAttribute was removed because this functionality is redundant in SYCL.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cublasLtMatmulPreferenceSetAttribute was removed because this functionality is redundant in SYCL.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cublasLtMatmulPreferenceDestroy was removed because this functionality is redundant in SYCL.
  // CHECK-NEXT: */
  cublasLtMatmulPreference_t matmulPreference;
  cublasLtMatmulPreferenceCreate(&matmulPreference);
  void *buf3;
  size_t sizeInBytes3;
  size_t *sizeWritten3;
  cublasLtMatmulPreferenceGetAttribute(matmulPreference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, buf3, sizeInBytes3, sizeWritten3);
  cublasLtMatmulPreferenceSetAttribute(matmulPreference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, buf3, sizeInBytes3);
  cublasLtMatmulPreferenceDestroy(matmulPreference);

  cublasLtMatrixLayout_t Adesc;
  cublasLtMatrixLayout_t Bdesc;
  cublasLtMatrixLayout_t Cdesc;
  cublasLtMatrixLayout_t Ddesc;

  // CHECK: int requestedAlgoCount = 1;
  // CHECK-NEXT: int heuristicResultsArray;
  // CHECK-NEXT: int returnAlgoCount;
  // CHECK-NEXT: returnAlgoCount = 1;
  int requestedAlgoCount = 1;
  cublasLtMatmulHeuristicResult_t heuristicResultsArray;
  int returnAlgoCount;
  cublasLtMatmulAlgoGetHeuristic(ltHandle, matmulDesc, Adesc, Bdesc, Cdesc, Ddesc, matmulPreference, requestedAlgoCount, &heuristicResultsArray, &returnAlgoCount);
}

void foo2() {
  // CHECK: dpct::blas_gemm::experimental::descriptor_ptr lightHandle;
  // CHECK-NEXT: dpct::blas_gemm::experimental::matmul_desc_ptr computeDesc;
  // CHECK-NEXT: const void *alpha;
  // CHECK-NEXT: const void *A;
  // CHECK-NEXT: dpct::blas_gemm::experimental::matrix_layout_ptr Adesc;
  // CHECK-NEXT: const void *B;
  // CHECK-NEXT: dpct::blas_gemm::experimental::matrix_layout_ptr Bdesc;
  // CHECK-NEXT: const void *beta;
  // CHECK-NEXT: const void *C;
  // CHECK-NEXT: dpct::blas_gemm::experimental::matrix_layout_ptr Cdesc;
  // CHECK-NEXT: void *D;
  // CHECK-NEXT: dpct::blas_gemm::experimental::matrix_layout_ptr Ddesc;
  // CHECK-NEXT: const int *algo;
  // CHECK-NEXT: void *workspace;
  // CHECK-NEXT: size_t workspaceSizeInBytes;
  // CHECK-NEXT: dpct::queue_ptr stream;
  // CHECK-NEXT: dpct::blas_gemm::experimental::matmul(lightHandle, computeDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, D, Ddesc, stream);
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
  // CHECK: dpct::blas_gemm::experimental::order_t a;
  // CHECK-NEXT: a = dpct::blas_gemm::experimental::order_t::col;
  // CHECK-NEXT: a = dpct::blas_gemm::experimental::order_t::row;
  // CHECK-NEXT: a = dpct::blas_gemm::experimental::order_t::col32;
  // CHECK-NEXT: a = dpct::blas_gemm::experimental::order_t::col4_4r2_8c;
  // CHECK-NEXT: a = dpct::blas_gemm::experimental::order_t::col32_2r_4r4;
  cublasLtOrder_t a;
  a = CUBLASLT_ORDER_COL;
  a = CUBLASLT_ORDER_ROW;
  a = CUBLASLT_ORDER_COL32;
  a = CUBLASLT_ORDER_COL4_4R2_8C;
  a = CUBLASLT_ORDER_COL32_2R_4R4;
  // CHECK: dpct::blas_gemm::experimental::pointer_mode_t b;
  // CHECK-NEXT: b = dpct::blas_gemm::experimental::pointer_mode_t::host;
  // CHECK-NEXT: b = dpct::blas_gemm::experimental::pointer_mode_t::device;
  // CHECK-NEXT: b = dpct::blas_gemm::experimental::pointer_mode_t::device_vector;
  // CHECK-NEXT: b = dpct::blas_gemm::experimental::pointer_mode_t::alpha_device_vector_beta_zero;
  // CHECK-NEXT: b = dpct::blas_gemm::experimental::pointer_mode_t::alpha_device_vector_beta_host;
  cublasLtPointerMode_t b;
  b = CUBLASLT_POINTER_MODE_HOST;
  b = CUBLASLT_POINTER_MODE_DEVICE;
  b = CUBLASLT_POINTER_MODE_DEVICE_VECTOR;
  b = CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO;
  b = CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST;
  // CHECK: dpct::blas_gemm::experimental::matrix_layout_t::attribute c;
  // CHECK-NEXT: c = dpct::blas_gemm::experimental::matrix_layout_t::attribute::type;
  // CHECK-NEXT: c = dpct::blas_gemm::experimental::matrix_layout_t::attribute::order;
  // CHECK-NEXT: c = dpct::blas_gemm::experimental::matrix_layout_t::attribute::rows;
  // CHECK-NEXT: c = dpct::blas_gemm::experimental::matrix_layout_t::attribute::cols;
  // CHECK-NEXT: c = dpct::blas_gemm::experimental::matrix_layout_t::attribute::ld;
  cublasLtMatrixLayoutAttribute_t c;
  c = CUBLASLT_MATRIX_LAYOUT_TYPE;
  c = CUBLASLT_MATRIX_LAYOUT_ORDER;
  c = CUBLASLT_MATRIX_LAYOUT_ROWS;
  c = CUBLASLT_MATRIX_LAYOUT_COLS;
  c = CUBLASLT_MATRIX_LAYOUT_LD;
  // CHECK: dpct::blas_gemm::experimental::matmul_desc_t::attribute d;
  // CHECK-NEXT: d = dpct::blas_gemm::experimental::matmul_desc_t::attribute::compute_type;
  // CHECK-NEXT: d = dpct::blas_gemm::experimental::matmul_desc_t::attribute::scale_type;
  // CHECK-NEXT: d = dpct::blas_gemm::experimental::matmul_desc_t::attribute::bias_type;
  // CHECK-NEXT: d = dpct::blas_gemm::experimental::matmul_desc_t::attribute::pointer_mode;
  // CHECK-NEXT: d = dpct::blas_gemm::experimental::matmul_desc_t::attribute::trans_a;
  // CHECK-NEXT: d = dpct::blas_gemm::experimental::matmul_desc_t::attribute::trans_b;
  // CHECK-NEXT: d = dpct::blas_gemm::experimental::matmul_desc_t::attribute::trans_c;
  // CHECK-NEXT: d = dpct::blas_gemm::experimental::matmul_desc_t::attribute::epilogue;
  // CHECK-NEXT: d = dpct::blas_gemm::experimental::matmul_desc_t::attribute::unsupport;
  // CHECK-NEXT: d = dpct::blas_gemm::experimental::matmul_desc_t::attribute::unsupport;
  // CHECK-NEXT: d = dpct::blas_gemm::experimental::matmul_desc_t::attribute::a_scale_pointer;
  // CHECK-NEXT: d = dpct::blas_gemm::experimental::matmul_desc_t::attribute::b_scale_pointer;
  // CHECK-NEXT: d = dpct::blas_gemm::experimental::matmul_desc_t::attribute::d_scale_pointer;
  // CHECK-NEXT: d = dpct::blas_gemm::experimental::matmul_desc_t::attribute::absmax_d_pointer;
  // CHECK-NEXT: d = dpct::blas_gemm::experimental::matmul_desc_t::attribute::unsupport;
  // CHECK-NEXT: d = dpct::blas_gemm::experimental::matmul_desc_t::attribute::unsupport;
  // CHECK-NEXT: d = dpct::blas_gemm::experimental::matmul_desc_t::attribute::unsupport;
  // CHECK-NEXT: d = dpct::blas_gemm::experimental::matmul_desc_t::attribute::unsupport;
  cublasLtMatmulDescAttributes_t d;
  d = CUBLASLT_MATMUL_DESC_COMPUTE_TYPE;
  d = CUBLASLT_MATMUL_DESC_SCALE_TYPE;
  d = CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE;
  d = CUBLASLT_MATMUL_DESC_POINTER_MODE;
  d = CUBLASLT_MATMUL_DESC_TRANSA;
  d = CUBLASLT_MATMUL_DESC_TRANSB;
  d = CUBLASLT_MATMUL_DESC_TRANSC;
  d = CUBLASLT_MATMUL_DESC_EPILOGUE;
  d = CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET;
  d = CUBLASLT_MATMUL_DESC_FAST_ACCUM;
  d = CUBLASLT_MATMUL_DESC_A_SCALE_POINTER;
  d = CUBLASLT_MATMUL_DESC_B_SCALE_POINTER;
  d = CUBLASLT_MATMUL_DESC_D_SCALE_POINTER;
  d = CUBLASLT_MATMUL_DESC_AMAX_D_POINTER;
  d = CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_ROWS;
  d = CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_COLS;
  d = CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_OUT_COUNTERS_POINTER;
  d = CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_IN_COUNTERS_POINTER;

  // CHECK: dpct::blas_gemm::experimental::epilogue_t e;
  // CHECK-NEXT: e = dpct::blas_gemm::experimental::epilogue_t::nop;
  // CHECK-NEXT: e = dpct::blas_gemm::experimental::epilogue_t::relu;
  cublasLtEpilogue_t e;
  e = CUBLASLT_EPILOGUE_DEFAULT;
  e = CUBLASLT_EPILOGUE_RELU;
}

void foo4() {
  // CHECK: dpct::blas_gemm::experimental::transform_desc_ptr transformDesc;
  // CHECK-NEXT: dpct::library_data_t scaleType;
  // CHECK-NEXT: transformDesc = new dpct::blas_gemm::experimental::transform_desc_t(scaleType);
  // CHECK-NEXT: oneapi::mkl::transpose opT = oneapi::mkl::transpose::trans;
  // CHECK-NEXT: size_t sizeWritten;
  // CHECK-NEXT: transformDesc->set_attribute(dpct::blas_gemm::experimental::transform_desc_t::attribute::trans_a, &opT);
  // CHECK-NEXT: transformDesc->get_attribute(dpct::blas_gemm::experimental::transform_desc_t::attribute::trans_a, &opT);
  // CHECK-NEXT: delete (transformDesc);
  cublasLtMatrixTransformDesc_t transformDesc;
  cudaDataType scaleType;
  cublasLtMatrixTransformDescCreate(&transformDesc, scaleType);
  cublasOperation_t opT = CUBLAS_OP_T;
  size_t sizeWritten;
  cublasLtMatrixTransformDescSetAttribute(transformDesc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opT, sizeof(opT));
  cublasLtMatrixTransformDescGetAttribute(transformDesc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opT, sizeof(opT), &sizeWritten);
  cublasLtMatrixTransformDescDestroy(transformDesc);

  // CHECK: dpct::blas_gemm::experimental::descriptor_ptr lightHandle;
  // CHECK-NEXT: const void *alpha;
  // CHECK-NEXT: const void *A;
  // CHECK-NEXT: dpct::blas_gemm::experimental::matrix_layout_ptr Adesc;
  // CHECK-NEXT: const void *beta;
  // CHECK-NEXT: const void *B;
  // CHECK-NEXT: dpct::blas_gemm::experimental::matrix_layout_ptr Bdesc;
  // CHECK-NEXT: void *C;
  // CHECK-NEXT: dpct::blas_gemm::experimental::matrix_layout_ptr Cdesc;
  // CHECK-NEXT: dpct::queue_ptr stream;
  // CHECK-NEXT: dpct::blas_gemm::experimental::matrix_transform(transformDesc, alpha, A, Adesc, beta, B, Bdesc, C, Cdesc, stream);
  cublasLtHandle_t lightHandle;
  const void *alpha;
  const void *A;
  cublasLtMatrixLayout_t Adesc;
  const void *beta;
  const void *B;
  cublasLtMatrixLayout_t Bdesc;
  void *C;
  cublasLtMatrixLayout_t Cdesc;
  cudaStream_t stream;
  cublasLtMatrixTransform(lightHandle, transformDesc, alpha, A, Adesc, beta, B, Bdesc, C, Cdesc, stream);
}

void foo5() {
  // CHECK: /*
  // CHECK-NEXT: DPCT1043:{{[0-9]+}}: The version-related API is different in SYCL. An initial code was generated, but you need to adjust it.
  // CHECK-NEXT: */
  // CHECK-NEXT: size_t ver = dpct::dnnl::get_version();
  size_t ver = cublasLtGetVersion();
}
