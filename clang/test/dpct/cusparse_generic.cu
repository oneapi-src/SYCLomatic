// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.4, v10.1
// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1
// RUN: dpct --format-range=none --out-root %T/cusparse_generic %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --input-file %T/cusparse_generic/cusparse_generic.dp.cpp --match-full-lines %s

#include "cusparse.h"

int main() {
  //CHECK:dpct::sparse::sparse_matrix_desc_t spMatDescr;
  //CHECK-NEXT:int64_t rows;
  //CHECK-NEXT:int64_t cols;
  //CHECK-NEXT:int64_t nnz;
  //CHECK-NEXT:void *csrRowOffsets;
  //CHECK-NEXT:void *csrColInd;
  //CHECK-NEXT:void *csrValues;
  //CHECK-NEXT:dpct::library_data_t csrRowOffsetsType;
  //CHECK-NEXT:dpct::library_data_t csrColIndType;
  //CHECK-NEXT:oneapi::mkl::index_base idxBase;
  //CHECK-NEXT:dpct::library_data_t valueType;
  //CHECK-NEXT:dpct::sparse::matrix_format format;
  cusparseSpMatDescr_t spMatDescr;
  int64_t rows;
  int64_t cols;
  int64_t nnz;
  void *csrRowOffsets;
  void *csrColInd;
  void *csrValues;
  cusparseIndexType_t csrRowOffsetsType;
  cusparseIndexType_t csrColIndType;
  cusparseIndexBase_t idxBase;
  cudaDataType valueType;
  cusparseFormat_t format;

  //CHECK:dpct::sparse::sparse_matrix_desc::create_csr(&spMatDescr, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues, csrRowOffsetsType, csrColIndType, idxBase, valueType);
  //CHECK-NEXT:dpct::sparse::sparse_matrix_desc::destroy(spMatDescr);
  //CHECK-NEXT:spMatDescr->get_desc(&rows, &cols, &nnz, &csrRowOffsets, &csrColInd, &csrValues, &csrRowOffsetsType, &csrColIndType, &idxBase, &valueType);
  //CHECK-NEXT:spMatDescr->get_format(&format);
  //CHECK-NEXT:spMatDescr->get_base(&idxBase);
  //CHECK-NEXT:spMatDescr->get_value(&csrValues);
  //CHECK-NEXT:spMatDescr->set_value(csrValues);
  //CHECK-NEXT:spMatDescr->set_pointers(csrRowOffsets, csrColInd, csrValues);
  //CHECK-NEXT:spMatDescr->get_size(&rows, &cols, &nnz);
  cusparseCreateCsr(&spMatDescr, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues, csrRowOffsetsType, csrColIndType, idxBase, valueType);
  cusparseDestroySpMat(spMatDescr);
  cusparseCsrGet(spMatDescr, &rows, &cols, &nnz, &csrRowOffsets, &csrColInd, &csrValues, &csrRowOffsetsType, &csrColIndType, &idxBase, &valueType);
  cusparseSpMatGetFormat(spMatDescr, &format);
  cusparseSpMatGetIndexBase(spMatDescr, &idxBase);
  cusparseSpMatGetValues(spMatDescr, &csrValues);
  cusparseSpMatSetValues(spMatDescr, csrValues);
  cusparseCsrSetPointers(spMatDescr, csrRowOffsets, csrColInd, csrValues);
  cusparseSpMatGetSize(spMatDescr, &rows, &cols, &nnz);

  //CHECK:void *data;
  //CHECK-NEXT:spMatDescr->get_attribute(dpct::sparse::matrix_attribute::uplo, &data, sizeof(oneapi::mkl::uplo));
  //CHECK-NEXT:spMatDescr->set_attribute(dpct::sparse::matrix_attribute::diag, data, sizeof(oneapi::mkl::diag));
  void *data;
  cusparseSpMatGetAttribute(spMatDescr, CUSPARSE_SPMAT_FILL_MODE, &data, sizeof(cusparseFillMode_t));
  cusparseSpMatSetAttribute(spMatDescr, CUSPARSE_SPMAT_DIAG_TYPE, data, sizeof(cusparseDiagType_t));

  //CHECK:std::shared_ptr<dpct::sparse::dense_matrix_desc> dnMatDescr;
  //CHECK-NEXT:int64_t ld;
  //CHECK-NEXT:oneapi::mkl::layout order;
  //CHECK-NEXT:void *values;
  cusparseDnMatDescr_t dnMatDescr;
  int64_t ld;
  cusparseOrder_t order;
  void *values;

  //CHECK:dnMatDescr = std::make_shared<dpct::sparse::dense_matrix_desc>(rows, cols, ld, values, valueType, order);
  //CHECK-NEXT:dnMatDescr.reset();
  //CHECK-NEXT:dnMatDescr->get_desc(&rows, &cols, &ld, &values, &valueType, &order);
  //CHECK-NEXT:values = dnMatDescr->_value;
  //CHECK-NEXT:dnMatDescr->_value = values;
  cusparseCreateDnMat(&dnMatDescr, rows, cols, ld, values, valueType, order);
  cusparseDestroyDnMat(dnMatDescr);
  cusparseDnMatGet(dnMatDescr, &rows, &cols, &ld, &values, &valueType, &order);
  cusparseDnMatGetValues(dnMatDescr, &values);
  cusparseDnMatSetValues(dnMatDescr, values);

  //CHECK:std::shared_ptr<dpct::sparse::dense_vector_desc> dnVecDescr;
  //CHECK-NEXT:int64_t size;
  cusparseDnVecDescr_t dnVecDescr;
  int64_t size;

  //CHECK:dnVecDescr = std::make_shared<dpct::sparse::dense_vector_desc>(size, values, valueType);
  //CHECK-NEXT:dnVecDescr.reset();
  //CHECK-NEXT:dnVecDescr->get_desc(&size, &values, &valueType);
  //CHECK-NEXT:values = dnVecDescr->_value;
  //CHECK-NEXT:dnVecDescr->_value = values;
  cusparseCreateDnVec(&dnVecDescr, size, values, valueType);
  cusparseDestroyDnVec(dnVecDescr);
  cusparseDnVecGet(dnVecDescr, &size, &values, &valueType);
  cusparseDnVecGetValues(dnVecDescr, &values);
  cusparseDnVecSetValues(dnVecDescr, values);
  return 0;
}
