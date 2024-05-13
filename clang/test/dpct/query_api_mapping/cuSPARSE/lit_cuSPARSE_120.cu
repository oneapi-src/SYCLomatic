// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-11.8
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6, v11.7, v11.8

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseCreateConstDnVec | FileCheck %s -check-prefix=cusparseCreateConstDnVec
// cusparseCreateConstDnVec: CUDA API:
// cusparseCreateConstDnVec-NEXT:   cusparseConstDnVecDescr_t desc;
// cusparseCreateConstDnVec-NEXT:   cusparseCreateConstDnVec(&desc /*cusparseConstDnVecDescr_t **/,
// cusparseCreateConstDnVec-NEXT:                            size /*int64_t*/, value /*const void **/,
// cusparseCreateConstDnVec-NEXT:                            value_type /*cudaDataType*/);
// cusparseCreateConstDnVec-NEXT: Is migrated to:
// cusparseCreateConstDnVec-NEXT:   std::shared_ptr<dpct::sparse::dense_vector_desc> desc;
// cusparseCreateConstDnVec-NEXT:   desc = std::make_shared<dpct::sparse::dense_vector_desc>(size, value, value_type);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseConstDnVecGet | FileCheck %s -check-prefix=cusparseConstDnVecGet
// cusparseConstDnVecGet: CUDA API:
// cusparseConstDnVecGet-NEXT:   int64_t size;
// cusparseConstDnVecGet-NEXT:   const void *value;
// cusparseConstDnVecGet-NEXT:   cudaDataType value_type;
// cusparseConstDnVecGet-NEXT:   cusparseConstDnVecGet(desc /*cusparseConstDnVecDescr_t*/, &size /*int64_t **/,
// cusparseConstDnVecGet-NEXT:                         &value /*const void ***/,
// cusparseConstDnVecGet-NEXT:                         &value_type /*cudaDataType **/);
// cusparseConstDnVecGet-NEXT: Is migrated to:
// cusparseConstDnVecGet-NEXT:   int64_t size;
// cusparseConstDnVecGet-NEXT:   const void *value;
// cusparseConstDnVecGet-NEXT:   dpct::library_data_t value_type;
// cusparseConstDnVecGet-NEXT:   desc->get_desc(&size, &value, &value_type);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseConstDnVecGetValues | FileCheck %s -check-prefix=cusparseConstDnVecGetValues
// cusparseConstDnVecGetValues: CUDA API:
// cusparseConstDnVecGetValues-NEXT:   const void *value;
// cusparseConstDnVecGetValues-NEXT:   cusparseConstDnVecGetValues(desc /*cusparseConstDnVecDescr_t*/,
// cusparseConstDnVecGetValues-NEXT:                               &value /*const void ***/);
// cusparseConstDnVecGetValues-NEXT: Is migrated to:
// cusparseConstDnVecGetValues-NEXT:   const void *value;
// cusparseConstDnVecGetValues-NEXT:   value = desc->get_value();
