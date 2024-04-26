// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3
// RUN: dpct --format-range=none --out-root %T/cusparse-after114 %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --input-file %T/cusparse-after114/cusparse-after114.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/cusparse-after114/cusparse-after114.dp.cpp -o %T/cusparse-after114/cusparse-after114.dp.o %}

#include <cusparse_v2.h>

void foo1() {
  //CHECK:int spsmDescr;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1026:{{[0-9]+}}: The call to cusparseSpSM_createDescr was removed because this functionality is redundant in SYCL.
  //CHECK-NEXT:*/
  cusparseSpSMDescr_t spsmDescr;
  cusparseSpSM_createDescr(&spsmDescr);

  cusparseHandle_t handle;
  cusparseOperation_t opA;
  cusparseOperation_t opB;
  const void *alpha;
  //CHECK:dpct::sparse::sparse_matrix_desc_t matA;
  //CHECK-NEXT:std::shared_ptr<dpct::sparse::dense_matrix_desc> matB;
  cusparseConstSpMatDescr_t matA;
  cusparseConstDnMatDescr_t matB;
  cusparseDnMatDescr_t matC;
  cudaDataType computeType;
  //CHECK:int alg;
  cusparseSpSMAlg_t alg;
  size_t bufferSize;
  void *externalBuffer;

  cusparseStatus_t status;
  //CHECK:status = DPCT_CHECK_ERROR(bufferSize = 0);
  //CHECK-NEXT:status = DPCT_CHECK_ERROR(dpct::sparse::spsm_optimize(*handle, opA, matA, matB, matC));
  //CHECK-NEXT:status = DPCT_CHECK_ERROR(dpct::sparse::spsm(*handle, opA, opB, alpha, matA, matB, matC, computeType));
  status = cusparseSpSM_bufferSize(handle, opA, opB, alpha, matA, matB, matC, computeType, alg, spsmDescr, &bufferSize);
  status = cusparseSpSM_analysis(handle, opA, opB, alpha, matA, matB, matC, computeType, alg, spsmDescr, externalBuffer);
  status = cusparseSpSM_solve(handle, opA, opB, alpha, matA, matB, matC, computeType, alg, spsmDescr);

  //CHECK:/*
  //CHECK-NEXT:DPCT1026:{{[0-9]+}}: The call to cusparseSpSM_destroyDescr was removed because this functionality is redundant in SYCL.
  //CHECK-NEXT:*/
  cusparseSpSM_destroyDescr(spsmDescr);
}
