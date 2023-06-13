// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.4, v10.1
// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1
// RUN: dpct --format-range=none --out-root %T/cusparse_generic_12 %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --input-file %T/cusparse_generic_12/cusparse_generic_12.dp.cpp --match-full-lines %s

#include "cusparse.h"

int main() {
  //CHECK:std::shared_ptr<dpct::sparse::dense_vector_desc> constDnVecDescr;
  //CHECK-NEXT:const void *c_values;
  //CHECK-NEXT:int64_t size;
  //CHECK-NEXT:void *values;
  //CHECK-NEXT:dpct::library_data_t valueType;
  //CHECK-NEXT:constDnVecDescr = std::make_shared<dpct::sparse::dense_vector_desc>(size, values, valueType);
  //CHECK-NEXT:constDnVecDescr->get_desc(&size, &c_values, &valueType);
  //CHECK-NEXT:c_values = constDnVecDescr->_value;
  cusparseConstDnVecDescr_t constDnVecDescr;
  const void *c_values;
  int64_t size;
  void *values;
  cudaDataType valueType;
  cusparseCreateConstDnVec(&constDnVecDescr, size, values, valueType);
  cusparseConstDnVecGet(constDnVecDescr, &size, &c_values, &valueType);
  cusparseConstDnVecGetValues(constDnVecDescr, &c_values);
  return 0;
}
