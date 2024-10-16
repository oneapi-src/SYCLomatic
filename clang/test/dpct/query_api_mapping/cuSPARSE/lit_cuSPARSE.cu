// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseSpMM_preprocess | FileCheck %s -check-prefix=cusparseSpMM_preprocess
// cusparseSpMM_preprocess: CUDA API:
// cusparseSpMM_preprocess-NEXT:   cusparseSpMM_preprocess(
// cusparseSpMM_preprocess-NEXT:       handle /*cusparseHandle_t*/, transa /*cusparseOperation_t*/,
// cusparseSpMM_preprocess-NEXT:       transb /*cusparseOperation_t*/, alpha /*const void **/,
// cusparseSpMM_preprocess-NEXT:       a /*cusparseSpMatDescr_t*/, b /*cusparseDnMatDescr_t*/,
// cusparseSpMM_preprocess-NEXT:       beta /*const void **/, c /*cusparseDnMatDescr_t*/,
// cusparseSpMM_preprocess-NEXT:       computetype /*cudaDataType*/, algo /*cusparseSpMMAlg_t*/,
// cusparseSpMM_preprocess-NEXT:       workspace /*void **/);
// cusparseSpMM_preprocess-NEXT:   The API is Removed.
// cusparseSpMM_preprocess-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseSpMM_bufferSize | FileCheck %s -check-prefix=cusparseSpMM_bufferSize
// cusparseSpMM_bufferSize: CUDA API:
// cusparseSpMM_bufferSize-NEXT:   cusparseSpMM_bufferSize(
// cusparseSpMM_bufferSize-NEXT:       handle /*cusparseHandle_t*/, transa /*cusparseOperation_t*/,
// cusparseSpMM_bufferSize-NEXT:       transb /*cusparseOperation_t*/, alpha /*const void **/,
// cusparseSpMM_bufferSize-NEXT:       a /*cusparseSpMatDescr_t*/, b /*cusparseDnMatDescr_t*/,
// cusparseSpMM_bufferSize-NEXT:       beta /*const void **/, c /*cusparseDnMatDescr_t*/,
// cusparseSpMM_bufferSize-NEXT:       computetype /*cudaDataType*/, algo /*cusparseSpMMAlg_t*/,
// cusparseSpMM_bufferSize-NEXT:       workspace_size /*size_t **/);
// cusparseSpMM_bufferSize-NEXT: Is migrated to:
// cusparseSpMM_bufferSize-NEXT:   *workspace_size = 0;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseSpMM | FileCheck %s -check-prefix=cusparseSpMM
// cusparseSpMM: CUDA API:
// cusparseSpMM-NEXT:   cusparseSpMM(handle /*cusparseHandle_t*/, transa /*cusparseOperation_t*/,
// cusparseSpMM-NEXT:                transb /*cusparseOperation_t*/, alpha /*const void **/,
// cusparseSpMM-NEXT:                a /*cusparseSpMatDescr_t*/, b /*cusparseDnMatDescr_t*/,
// cusparseSpMM-NEXT:                beta /*const void **/, c /*cusparseDnMatDescr_t*/,
// cusparseSpMM-NEXT:                computetype /*cudaDataType*/, algo /*cusparseSpMMAlg_t*/,
// cusparseSpMM-NEXT:                workspace /*void **/);
// cusparseSpMM-NEXT: Is migrated to:
// cusparseSpMM-NEXT:   dpct::sparse::spmm(handle->get_queue(), transa, transb, alpha, a, b, beta, c, computetype);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseCreate | FileCheck %s -check-prefix=cusparseCreate
// cusparseCreate: CUDA API:
// cusparseCreate-NEXT:   cusparseHandle_t handle;
// cusparseCreate-NEXT:   cusparseCreate(&handle /*cusparseHandle_t **/);
// cusparseCreate-NEXT: Is migrated to:
// cusparseCreate-NEXT:   dpct::sparse::descriptor_ptr handle;
// cusparseCreate-NEXT:   handle = new dpct::sparse::descriptor();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseDestroy | FileCheck %s -check-prefix=cusparseDestroy
// cusparseDestroy: CUDA API:
// cusparseDestroy-NEXT:   cusparseDestroy(handle /*cusparseHandle_t*/);
// cusparseDestroy-NEXT: Is migrated to:
// cusparseDestroy-NEXT:   delete (handle);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseCreateMatDescr | FileCheck %s -check-prefix=cusparseCreateMatDescr
// cusparseCreateMatDescr: CUDA API:
// cusparseCreateMatDescr-NEXT:   cusparseMatDescr_t desc;
// cusparseCreateMatDescr-NEXT:   cusparseCreateMatDescr(&desc /*cusparseMatDescr_t **/);
// cusparseCreateMatDescr-NEXT: Is migrated to:
// cusparseCreateMatDescr-NEXT:   std::shared_ptr<dpct::sparse::matrix_info> desc;
// cusparseCreateMatDescr-NEXT:   desc = std::make_shared<dpct::sparse::matrix_info>();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseDestroyMatDescr | FileCheck %s -check-prefix=cusparseDestroyMatDescr
// cusparseDestroyMatDescr: CUDA API:
// cusparseDestroyMatDescr-NEXT:   cusparseDestroyMatDescr(desc /*cusparseMatDescr_t*/);
// cusparseDestroyMatDescr-NEXT: The API is Removed.

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseSpMV_bufferSize | FileCheck %s -check-prefix=cusparseSpMV_bufferSize
// cusparseSpMV_bufferSize: CUDA API:
// cusparseSpMV_bufferSize-NEXT:   size_t buffer_size;
// cusparseSpMV_bufferSize-NEXT:   cusparseSpMV_bufferSize(
// cusparseSpMV_bufferSize-NEXT:       handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/,
// cusparseSpMV_bufferSize-NEXT:       alpha /*const void **/, mat_a /*cusparseSpMatDescr_t*/,
// cusparseSpMV_bufferSize-NEXT:       vec_x /*cusparseDnVecDescr_t*/, beta /*const void **/,
// cusparseSpMV_bufferSize-NEXT:       vec_y /*cusparseDnVecDescr_t*/, compute_type /*cudaDataType*/,
// cusparseSpMV_bufferSize-NEXT:       alg /*cusparseSpMVAlg_t*/, &buffer_size /*size_t **/);
// cusparseSpMV_bufferSize-NEXT: Is migrated to:
// cusparseSpMV_bufferSize-NEXT:   size_t buffer_size;
// cusparseSpMV_bufferSize-NEXT:   buffer_size = 0;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseSpMV | FileCheck %s -check-prefix=cusparseSpMV
// cusparseSpMV: CUDA API:
// cusparseSpMV-NEXT:   cusparseSpMV(handle /*cusparseHandle_t*/, trans /*cusparseOperation_t*/,
// cusparseSpMV-NEXT:                alpha /*const void **/, mat_a /*cusparseSpMatDescr_t*/,
// cusparseSpMV-NEXT:                vec_x /*cusparseDnVecDescr_t*/, beta /*const void **/,
// cusparseSpMV-NEXT:                vec_y /*cusparseDnVecDescr_t*/, compute_type /*cudaDataType*/,
// cusparseSpMV-NEXT:                alg /*cusparseSpMVAlg_t*/, buffer /*void **/);
// cusparseSpMV-NEXT: Is migrated to:
// cusparseSpMV-NEXT:   dpct::sparse::spmv(handle->get_queue(), trans, alpha, mat_a, vec_x, beta, vec_y, compute_type);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseCsr2cscEx2_bufferSize | FileCheck %s -check-prefix=cusparseCsr2cscEx2_bufferSize
// cusparseCsr2cscEx2_bufferSize: CUDA API:
// cusparseCsr2cscEx2_bufferSize-NEXT:   size_t buffer_size;
// cusparseCsr2cscEx2_bufferSize-NEXT:   cusparseCsr2cscEx2_bufferSize(
// cusparseCsr2cscEx2_bufferSize-NEXT:       handle /*cusparseHandle_t*/, m /*int*/, n /*int*/, nnz /*int*/,
// cusparseCsr2cscEx2_bufferSize-NEXT:       csr_value /*const void **/, row_ptr /*const int **/,
// cusparseCsr2cscEx2_bufferSize-NEXT:       col_idx /*const int **/, csc_value /*void **/, col_ptr /*int **/,
// cusparseCsr2cscEx2_bufferSize-NEXT:       row_ind /*int **/, value_type /*cudaDataType*/, act /*cusparseAction_t*/,
// cusparseCsr2cscEx2_bufferSize-NEXT:       base /*cusparseIndexBase_t*/, alg /*cusparseCsr2CscAlg_t*/,
// cusparseCsr2cscEx2_bufferSize-NEXT:       &buffer_size /*size_t **/);
// cusparseCsr2cscEx2_bufferSize-NEXT: Is migrated to:
// cusparseCsr2cscEx2_bufferSize-NEXT:   size_t buffer_size;
// cusparseCsr2cscEx2_bufferSize-NEXT:   buffer_size = 0;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseCsr2cscEx2 | FileCheck %s -check-prefix=cusparseCsr2cscEx2
// cusparseCsr2cscEx2: CUDA API:
// cusparseCsr2cscEx2-NEXT:   cusparseCsr2cscEx2(handle /*cusparseHandle_t*/, m /*int*/, n /*int*/,
// cusparseCsr2cscEx2-NEXT:                      nnz /*int*/, csr_value /*const void **/,
// cusparseCsr2cscEx2-NEXT:                      row_ptr /*const int **/, col_idx /*const int **/,
// cusparseCsr2cscEx2-NEXT:                      csc_value /*void **/, col_ptr /*int **/, row_ind /*int **/,
// cusparseCsr2cscEx2-NEXT:                      value_type /*cudaDataType*/, act /*cusparseAction_t*/,
// cusparseCsr2cscEx2-NEXT:                      base /*cusparseIndexBase_t*/, alg /*cusparseCsr2CscAlg_t*/,
// cusparseCsr2cscEx2-NEXT:                      buffer /*void **/);
// cusparseCsr2cscEx2-NEXT: Is migrated to:
// cusparseCsr2cscEx2-NEXT:   dpct::sparse::csr2csc(handle->get_queue(), m, n, nnz, csr_value, row_ptr, col_idx, csc_value, col_ptr, row_ind, value_type, act, base);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseCreateDnVec | FileCheck %s -check-prefix=cusparseCreateDnVec
// cusparseCreateDnVec: CUDA API:
// cusparseCreateDnVec-NEXT:   cusparseDnVecDescr_t desc;
// cusparseCreateDnVec-NEXT:   cusparseCreateDnVec(&desc /*cusparseDnVecDescr_t **/, size /*int64_t*/,
// cusparseCreateDnVec-NEXT:                       value /*void **/, value_type /*cudaDataType*/);
// cusparseCreateDnVec-NEXT: Is migrated to:
// cusparseCreateDnVec-NEXT:   std::shared_ptr<dpct::sparse::dense_vector_desc> desc;
// cusparseCreateDnVec-NEXT:   desc = std::make_shared<dpct::sparse::dense_vector_desc>(size, value, value_type);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseDestroyDnVec | FileCheck %s -check-prefix=cusparseDestroyDnVec
// cusparseDestroyDnVec: CUDA API:
// cusparseDestroyDnVec-NEXT:   cusparseDestroyDnVec(desc /*cusparseDnVecDescr_t*/);
// cusparseDestroyDnVec-NEXT: Is migrated to:
// cusparseDestroyDnVec-NEXT:   desc.reset();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseDnVecGet | FileCheck %s -check-prefix=cusparseDnVecGet
// cusparseDnVecGet: CUDA API:
// cusparseDnVecGet-NEXT:   int64_t size;
// cusparseDnVecGet-NEXT:   void *value;
// cusparseDnVecGet-NEXT:   cudaDataType value_type;
// cusparseDnVecGet-NEXT:   cusparseDnVecGet(desc /*cusparseDnVecDescr_t*/, &size /*int64_t **/,
// cusparseDnVecGet-NEXT:                    &value /*void ***/, &value_type /*cudaDataType **/);
// cusparseDnVecGet-NEXT: Is migrated to:
// cusparseDnVecGet-NEXT:   int64_t size;
// cusparseDnVecGet-NEXT:   void *value;
// cusparseDnVecGet-NEXT:   dpct::library_data_t value_type;
// cusparseDnVecGet-NEXT:   desc->get_desc(&size, &value, &value_type);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseDnVecGetValues | FileCheck %s -check-prefix=cusparseDnVecGetValues
// cusparseDnVecGetValues: CUDA API:
// cusparseDnVecGetValues-NEXT:   void *value;
// cusparseDnVecGetValues-NEXT:   cusparseDnVecGetValues(desc /*cusparseDnVecDescr_t*/, &value /*void ***/);
// cusparseDnVecGetValues-NEXT: Is migrated to:
// cusparseDnVecGetValues-NEXT:   void *value;
// cusparseDnVecGetValues-NEXT:   value = desc->get_value();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseDnVecSetValues | FileCheck %s -check-prefix=cusparseDnVecSetValues
// cusparseDnVecSetValues: CUDA API:
// cusparseDnVecSetValues-NEXT:   cusparseDnVecSetValues(desc /*cusparseDnVecDescr_t*/, value /*void **/);
// cusparseDnVecSetValues-NEXT: Is migrated to:
// cusparseDnVecSetValues-NEXT:   desc->set_value(value);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseCreateDnMat | FileCheck %s -check-prefix=cusparseCreateDnMat
// cusparseCreateDnMat: CUDA API:
// cusparseCreateDnMat-NEXT:   cusparseDnMatDescr_t desc;
// cusparseCreateDnMat-NEXT:   cusparseCreateDnMat(&desc /*cusparseDnMatDescr_t **/, rows /*int64_t*/,
// cusparseCreateDnMat-NEXT:                       cols /*int64_t*/, ld /*int64_t*/, value /*void **/,
// cusparseCreateDnMat-NEXT:                       value_type /*cudaDataType*/, order /*cusparseOrder_t*/);
// cusparseCreateDnMat-NEXT: Is migrated to:
// cusparseCreateDnMat-NEXT:   std::shared_ptr<dpct::sparse::dense_matrix_desc> desc;
// cusparseCreateDnMat-NEXT:   desc = std::make_shared<dpct::sparse::dense_matrix_desc>(rows, cols, ld, value, value_type, order);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseDestroyDnMat | FileCheck %s -check-prefix=cusparseDestroyDnMat
// cusparseDestroyDnMat: CUDA API:
// cusparseDestroyDnMat-NEXT:   cusparseDestroyDnMat(desc /*cusparseDnMatDescr_t*/);
// cusparseDestroyDnMat-NEXT: Is migrated to:
// cusparseDestroyDnMat-NEXT:   desc.reset();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseDnMatGet | FileCheck %s -check-prefix=cusparseDnMatGet
// cusparseDnMatGet: CUDA API:
// cusparseDnMatGet-NEXT:   int64_t rows;
// cusparseDnMatGet-NEXT:   int64_t cols;
// cusparseDnMatGet-NEXT:   int64_t ld;
// cusparseDnMatGet-NEXT:   void *value;
// cusparseDnMatGet-NEXT:   cudaDataType value_type;
// cusparseDnMatGet-NEXT:   cusparseOrder_t order;
// cusparseDnMatGet-NEXT:   cusparseDnMatGet(desc /*cusparseDnMatDescr_t*/, &rows /*int64_t **/,
// cusparseDnMatGet-NEXT:                    &cols /*int64_t **/, &ld /*int64_t **/, &value /*void ***/,
// cusparseDnMatGet-NEXT:                    &value_type /*cudaDataType **/,
// cusparseDnMatGet-NEXT:                    &order /*cusparseOrder_t **/);
// cusparseDnMatGet-NEXT: Is migrated to:
// cusparseDnMatGet-NEXT:   int64_t rows;
// cusparseDnMatGet-NEXT:   int64_t cols;
// cusparseDnMatGet-NEXT:   int64_t ld;
// cusparseDnMatGet-NEXT:   void *value;
// cusparseDnMatGet-NEXT:   dpct::library_data_t value_type;
// cusparseDnMatGet-NEXT:   oneapi::mkl::layout order;
// cusparseDnMatGet-NEXT:   desc->get_desc(&rows, &cols, &ld, &value, &value_type, &order);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseDnMatGetValues | FileCheck %s -check-prefix=cusparseDnMatGetValues
// cusparseDnMatGetValues: CUDA API:
// cusparseDnMatGetValues-NEXT:   void *value;
// cusparseDnMatGetValues-NEXT:   cusparseDnMatGetValues(desc /*cusparseDnMatDescr_t*/, &value /*void ***/);
// cusparseDnMatGetValues-NEXT: Is migrated to:
// cusparseDnMatGetValues-NEXT:   void *value;
// cusparseDnMatGetValues-NEXT:   value = desc->get_value();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseDnMatSetValues | FileCheck %s -check-prefix=cusparseDnMatSetValues
// cusparseDnMatSetValues: CUDA API:
// cusparseDnMatSetValues-NEXT:   cusparseDnMatSetValues(desc /*cusparseDnMatDescr_t*/, value /*void **/);
// cusparseDnMatSetValues-NEXT: Is migrated to:
// cusparseDnMatSetValues-NEXT:   desc->set_value(value);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseGetErrorName | FileCheck %s -check-prefix=cusparseGetErrorName
// cusparseGetErrorName: CUDA API:
// cusparseGetErrorName-NEXT:   const char *Name = cusparseGetErrorName(status /*cusparseStatus_t*/);
// cusparseGetErrorName-NEXT: Is migrated to:
// cusparseGetErrorName-NEXT:   const char *Name = dpct::get_error_string_dummy(status);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseGetErrorString | FileCheck %s -check-prefix=cusparseGetErrorString
// cusparseGetErrorString: CUDA API:
// cusparseGetErrorString-NEXT:   const char *Str = cusparseGetErrorString(status /*cusparseStatus_t*/);
// cusparseGetErrorString-NEXT: Is migrated to:
// cusparseGetErrorString-NEXT:   const char *Str = dpct::get_error_string_dummy(status);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseGetProperty | FileCheck %s -check-prefix=cusparseGetProperty
// cusparseGetProperty: CUDA API:
// cusparseGetProperty-NEXT:   int value;
// cusparseGetProperty-NEXT:   cusparseGetProperty(type /*libraryPropertyType*/, &value /*int **/);
// cusparseGetProperty-NEXT: Is migrated to:
// cusparseGetProperty-NEXT:   int value;
// cusparseGetProperty-NEXT:   dpct::mkl_get_version(type, &value);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseGetMatDiagType | FileCheck %s -check-prefix=cusparseGetMatDiagType
// cusparseGetMatDiagType: CUDA API:
// cusparseGetMatDiagType-NEXT:   cusparseDiagType_t diag = cusparseGetMatDiagType(desc /*cusparseMatDescr_t*/);
// cusparseGetMatDiagType-NEXT: Is migrated to:
// cusparseGetMatDiagType-NEXT:   oneapi::mkl::diag diag = desc->get_diag();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseGetMatFillMode | FileCheck %s -check-prefix=cusparseGetMatFillMode
// cusparseGetMatFillMode: CUDA API:
// cusparseGetMatFillMode-NEXT:   cusparseFillMode_t uplo = cusparseGetMatFillMode(desc /*cusparseMatDescr_t*/);
// cusparseGetMatFillMode-NEXT: Is migrated to:
// cusparseGetMatFillMode-NEXT:   oneapi::mkl::uplo uplo = desc->get_uplo();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseGetMatIndexBase | FileCheck %s -check-prefix=cusparseGetMatIndexBase
// cusparseGetMatIndexBase: CUDA API:
// cusparseGetMatIndexBase-NEXT:   cusparseIndexBase_t base =
// cusparseGetMatIndexBase-NEXT:       cusparseGetMatIndexBase(desc /*cusparseMatDescr_t*/);
// cusparseGetMatIndexBase-NEXT: Is migrated to:
// cusparseGetMatIndexBase-NEXT:   oneapi::mkl::index_base base =
// cusparseGetMatIndexBase-NEXT:       desc->get_index_base();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseGetMatType | FileCheck %s -check-prefix=cusparseGetMatType
// cusparseGetMatType: CUDA API:
// cusparseGetMatType-NEXT:   cusparseMatrixType_t mat_type =
// cusparseGetMatType-NEXT:       cusparseGetMatType(desc /*cusparseMatDescr_t*/);
// cusparseGetMatType-NEXT: Is migrated to:
// cusparseGetMatType-NEXT:   dpct::sparse::matrix_info::matrix_type mat_type =
// cusparseGetMatType-NEXT:       desc->get_matrix_type();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseGetPointerMode | FileCheck %s -check-prefix=cusparseGetPointerMode
// cusparseGetPointerMode: CUDA API:
// cusparseGetPointerMode-NEXT:   cusparseGetPointerMode(handle /*cusparseHandle_t*/,
// cusparseGetPointerMode-NEXT:                          mode /*cusparsePointerMode_t **/);
// cusparseGetPointerMode-NEXT: The API is Removed.

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseGetStream | FileCheck %s -check-prefix=cusparseGetStream
// cusparseGetStream: CUDA API:
// cusparseGetStream-NEXT:   cudaStream_t s;
// cusparseGetStream-NEXT:   cusparseGetStream(handle /*cusparseHandle_t*/, &s /*cudaStream_t **/);
// cusparseGetStream-NEXT: Is migrated to:
// cusparseGetStream-NEXT:   dpct::queue_ptr s;
// cusparseGetStream-NEXT:   s = &(handle->get_queue());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseSetMatDiagType | FileCheck %s -check-prefix=cusparseSetMatDiagType
// cusparseSetMatDiagType: CUDA API:
// cusparseSetMatDiagType-NEXT:   cusparseSetMatDiagType(desc /*cusparseMatDescr_t*/,
// cusparseSetMatDiagType-NEXT:                          diag /*cusparseDiagType_t*/);
// cusparseSetMatDiagType-NEXT: Is migrated to:
// cusparseSetMatDiagType-NEXT:   desc->set_diag(diag);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseSetMatFillMode | FileCheck %s -check-prefix=cusparseSetMatFillMode
// cusparseSetMatFillMode: CUDA API:
// cusparseSetMatFillMode-NEXT:   cusparseSetMatFillMode(desc /*cusparseMatDescr_t*/,
// cusparseSetMatFillMode-NEXT:                          uplo /*cusparseFillMode_t*/);
// cusparseSetMatFillMode-NEXT: Is migrated to:
// cusparseSetMatFillMode-NEXT:   desc->set_uplo(uplo);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseSetMatIndexBase | FileCheck %s -check-prefix=cusparseSetMatIndexBase
// cusparseSetMatIndexBase: CUDA API:
// cusparseSetMatIndexBase-NEXT:   cusparseSetMatIndexBase(desc /*cusparseMatDescr_t*/,
// cusparseSetMatIndexBase-NEXT:                           base /*cusparseIndexBase_t*/);
// cusparseSetMatIndexBase-NEXT: Is migrated to:
// cusparseSetMatIndexBase-NEXT:   desc->set_index_base(base);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseSetMatType | FileCheck %s -check-prefix=cusparseSetMatType
// cusparseSetMatType: CUDA API:
// cusparseSetMatType-NEXT:   cusparseSetMatType(desc /*cusparseMatDescr_t*/,
// cusparseSetMatType-NEXT:                      mat_type /*cusparseMatrixType_t*/);
// cusparseSetMatType-NEXT: Is migrated to:
// cusparseSetMatType-NEXT:   desc->set_matrix_type(mat_type);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseSetPointerMode | FileCheck %s -check-prefix=cusparseSetPointerMode
// cusparseSetPointerMode: CUDA API:
// cusparseSetPointerMode-NEXT:   cusparseSetPointerMode(handle /*cusparseHandle_t*/,
// cusparseSetPointerMode-NEXT:                          mode /*cusparsePointerMode_t*/);
// cusparseSetPointerMode-NEXT: The API is Removed.

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseSetStream | FileCheck %s -check-prefix=cusparseSetStream
// cusparseSetStream: CUDA API:
// cusparseSetStream-NEXT:   cusparseSetStream(handle /*cusparseHandle_t*/, s /*cudaStream_t*/);
// cusparseSetStream-NEXT: Is migrated to:
// cusparseSetStream-NEXT:   handle->set_queue(s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseSpMatGetFormat | FileCheck %s -check-prefix=cusparseSpMatGetFormat
// cusparseSpMatGetFormat: CUDA API:
// cusparseSpMatGetFormat-NEXT:   cusparseFormat_t format;
// cusparseSpMatGetFormat-NEXT:   cusparseSpMatGetFormat(desc /*cusparseSpMatDescr_t*/,
// cusparseSpMatGetFormat-NEXT:                          &format /*cusparseFormat_t **/);
// cusparseSpMatGetFormat-NEXT: Is migrated to:
// cusparseSpMatGetFormat-NEXT:   dpct::sparse::matrix_format format;
// cusparseSpMatGetFormat-NEXT:   desc->get_format(&format);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseSpMatGetIndexBase | FileCheck %s -check-prefix=cusparseSpMatGetIndexBase
// cusparseSpMatGetIndexBase: CUDA API:
// cusparseSpMatGetIndexBase-NEXT:   cusparseIndexBase_t base;
// cusparseSpMatGetIndexBase-NEXT:   cusparseSpMatGetIndexBase(desc /*cusparseSpMatDescr_t*/,
// cusparseSpMatGetIndexBase-NEXT:                             &base /*cusparseIndexBase_t **/);
// cusparseSpMatGetIndexBase-NEXT: Is migrated to:
// cusparseSpMatGetIndexBase-NEXT:   oneapi::mkl::index_base base;
// cusparseSpMatGetIndexBase-NEXT:   desc->get_base(&base);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseSpMatGetValues | FileCheck %s -check-prefix=cusparseSpMatGetValues
// cusparseSpMatGetValues: CUDA API:
// cusparseSpMatGetValues-NEXT:   void *value;
// cusparseSpMatGetValues-NEXT:   cusparseSpMatGetValues(desc /*cusparseSpMatDescr_t*/, &value /*void ***/);
// cusparseSpMatGetValues-NEXT: Is migrated to:
// cusparseSpMatGetValues-NEXT:   void *value;
// cusparseSpMatGetValues-NEXT:   desc->get_value(&value);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseSpMatSetValues | FileCheck %s -check-prefix=cusparseSpMatSetValues
// cusparseSpMatSetValues: CUDA API:
// cusparseSpMatSetValues-NEXT:   cusparseSpMatSetValues(desc /*cusparseSpMatDescr_t*/, value /*void **/);
// cusparseSpMatSetValues-NEXT: Is migrated to:
// cusparseSpMatSetValues-NEXT:   desc->set_value(value);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseSpMatGetSize | FileCheck %s -check-prefix=cusparseSpMatGetSize
// cusparseSpMatGetSize: CUDA API:
// cusparseSpMatGetSize-NEXT:   int64_t rows;
// cusparseSpMatGetSize-NEXT:   int64_t cols;
// cusparseSpMatGetSize-NEXT:   int64_t nnz;
// cusparseSpMatGetSize-NEXT:   cusparseSpMatGetSize(desc /*cusparseSpMatDescr_t*/, &rows /*int64_t **/,
// cusparseSpMatGetSize-NEXT:                        &cols /*int64_t **/, &nnz /*int64_t **/);
// cusparseSpMatGetSize-NEXT: Is migrated to:
// cusparseSpMatGetSize-NEXT:   int64_t rows;
// cusparseSpMatGetSize-NEXT:   int64_t cols;
// cusparseSpMatGetSize-NEXT:   int64_t nnz;
// cusparseSpMatGetSize-NEXT:   desc->get_size(&rows, &cols, &nnz);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseSpGEMM_compute | FileCheck %s -check-prefix=cusparseSpGEMM_compute
// cusparseSpGEMM_compute: CUDA API:
// cusparseSpGEMM_compute-NEXT:   cusparseSpGEMM_compute(
// cusparseSpGEMM_compute-NEXT:       handle /*cusparseHandle_t*/, op_a /*cusparseOperation_t*/,
// cusparseSpGEMM_compute-NEXT:       op_b /*cusparseOperation_t*/, alpha /*const void **/,
// cusparseSpGEMM_compute-NEXT:       mat_a /*cusparseSpMatDescr_t*/, mat_b /*cusparseSpMatDescr_t*/,
// cusparseSpGEMM_compute-NEXT:       beta /*const void **/, mat_c /*cusparseSpMatDescr_t*/,
// cusparseSpGEMM_compute-NEXT:       compute_type /*cudaDataType*/, alg /*cusparseSpGEMMAlg_t*/,
// cusparseSpGEMM_compute-NEXT:       desc /*cusparseSpGEMMDescr_t*/, buffer_size /*size_t **/,
// cusparseSpGEMM_compute-NEXT:       buffer /*void **/);
// cusparseSpGEMM_compute-NEXT: Is migrated to:
// cusparseSpGEMM_compute-NEXT:   dpct::sparse::spgemm_compute(handle->get_queue(), op_a, op_b, alpha, mat_a, mat_b, beta, mat_c, desc, buffer_size, buffer);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseSpGEMM_copy | FileCheck %s -check-prefix=cusparseSpGEMM_copy
// cusparseSpGEMM_copy: CUDA API:
// cusparseSpGEMM_copy-NEXT:   cusparseSpGEMM_copy(
// cusparseSpGEMM_copy-NEXT:       handle /*cusparseHandle_t*/, op_a /*cusparseOperation_t*/,
// cusparseSpGEMM_copy-NEXT:       op_b /*cusparseOperation_t*/, alpha /*const void **/,
// cusparseSpGEMM_copy-NEXT:       mat_a /*cusparseSpMatDescr_t*/, mat_b /*cusparseSpMatDescr_t*/,
// cusparseSpGEMM_copy-NEXT:       beta /*const void **/, mat_c /*cusparseSpMatDescr_t*/,
// cusparseSpGEMM_copy-NEXT:       compute_type /*cudaDataType*/, alg /*cusparseSpGEMMAlg_t*/,
// cusparseSpGEMM_copy-NEXT:       desc /*cusparseSpGEMMDescr_t*/);
// cusparseSpGEMM_copy-NEXT: Is migrated to:
// cusparseSpGEMM_copy-NEXT:   dpct::sparse::spgemm_finalize(handle->get_queue(), op_a, op_b, alpha, mat_a, mat_b, beta, mat_c, desc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseSpGEMM_createDescr | FileCheck %s -check-prefix=cusparseSpGEMM_createDescr
// cusparseSpGEMM_createDescr: CUDA API:
// cusparseSpGEMM_createDescr-NEXT:   cusparseSpGEMMDescr_t desc;
// cusparseSpGEMM_createDescr-NEXT:   cusparseSpGEMM_createDescr(&desc /*cusparseSpGEMMDescr_t **/);
// cusparseSpGEMM_createDescr-NEXT: Is migrated to:
// cusparseSpGEMM_createDescr-NEXT:   oneapi::mkl::sparse::matmat_descr_t desc;
// cusparseSpGEMM_createDescr-NEXT:   oneapi::mkl::sparse::init_matmat_descr(&desc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseSpGEMM_destroyDescr | FileCheck %s -check-prefix=cusparseSpGEMM_destroyDescr
// cusparseSpGEMM_destroyDescr: CUDA API:
// cusparseSpGEMM_destroyDescr-NEXT:   cusparseSpGEMM_destroyDescr(desc /*cusparseSpGEMMDescr_t*/);
// cusparseSpGEMM_destroyDescr-NEXT: Is migrated to:
// cusparseSpGEMM_destroyDescr-NEXT:   oneapi::mkl::sparse::release_matmat_descr(&desc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseSpGEMM_workEstimation | FileCheck %s -check-prefix=cusparseSpGEMM_workEstimation
// cusparseSpGEMM_workEstimation: CUDA API:
// cusparseSpGEMM_workEstimation-NEXT:   cusparseSpGEMM_workEstimation(
// cusparseSpGEMM_workEstimation-NEXT:       handle /*cusparseHandle_t*/, op_a /*cusparseOperation_t*/,
// cusparseSpGEMM_workEstimation-NEXT:       op_b /*cusparseOperation_t*/, alpha /*const void **/,
// cusparseSpGEMM_workEstimation-NEXT:       mat_a /*cusparseSpMatDescr_t*/, mat_b /*cusparseSpMatDescr_t*/,
// cusparseSpGEMM_workEstimation-NEXT:       beta /*const void **/, mat_c /*cusparseSpMatDescr_t*/,
// cusparseSpGEMM_workEstimation-NEXT:       compute_type /*cudaDataType*/, alg /*cusparseSpGEMMAlg_t*/,
// cusparseSpGEMM_workEstimation-NEXT:       desc /*cusparseSpGEMMDescr_t*/, buffer_size /*size_t **/,
// cusparseSpGEMM_workEstimation-NEXT:       buffer /*void **/);
// cusparseSpGEMM_workEstimation-NEXT: Is migrated to:
// cusparseSpGEMM_workEstimation-NEXT:   dpct::sparse::spgemm_work_estimation(handle->get_queue(), op_a, op_b, alpha, mat_a, mat_b, beta, mat_c, desc, buffer_size, buffer);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseCreateCsr | FileCheck %s -check-prefix=cusparseCreateCsr
// cusparseCreateCsr: CUDA API:
// cusparseCreateCsr-NEXT:   cusparseSpMatDescr_t desc;
// cusparseCreateCsr-NEXT:   cusparseCreateCsr(&desc /*cusparseSpMatDescr_t **/, rows /*int64_t*/,
// cusparseCreateCsr-NEXT:                     cols /*int64_t*/, nnz /*int64_t*/, row_ptr /*void **/,
// cusparseCreateCsr-NEXT:                     col_ind /*void **/, value /*void **/,
// cusparseCreateCsr-NEXT:                     row_ptr_type /*cusparseIndexType_t*/,
// cusparseCreateCsr-NEXT:                     col_ind_type /*cusparseIndexType_t*/,
// cusparseCreateCsr-NEXT:                     base /*cusparseIndexBase_t*/, value_type /*cudaDataType*/);
// cusparseCreateCsr-NEXT: Is migrated to:
// cusparseCreateCsr-NEXT:   dpct::sparse::sparse_matrix_desc_t desc;
// cusparseCreateCsr-NEXT:   desc = std::make_shared<dpct::sparse::sparse_matrix_desc>(rows, cols, nnz, row_ptr, col_ind, value, row_ptr_type, col_ind_type, base, value_type, dpct::sparse::matrix_format::csr);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseDestroySpMat | FileCheck %s -check-prefix=cusparseDestroySpMat
// cusparseDestroySpMat: CUDA API:
// cusparseDestroySpMat-NEXT:   cusparseDestroySpMat(desc /*cusparseSpMatDescr_t*/);
// cusparseDestroySpMat-NEXT: Is migrated to:
// cusparseDestroySpMat-NEXT:   desc.reset();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseCsrGet | FileCheck %s -check-prefix=cusparseCsrGet
// cusparseCsrGet: CUDA API:
// cusparseCsrGet-NEXT:   int64_t rows;
// cusparseCsrGet-NEXT:   int64_t cols;
// cusparseCsrGet-NEXT:   int64_t nnz;
// cusparseCsrGet-NEXT:   void *row_ptr;
// cusparseCsrGet-NEXT:   void *col_ind;
// cusparseCsrGet-NEXT:   void *value;
// cusparseCsrGet-NEXT:   cusparseIndexType_t row_ptr_type;
// cusparseCsrGet-NEXT:   cusparseIndexType_t col_ind_type;
// cusparseCsrGet-NEXT:   cusparseIndexBase_t base;
// cusparseCsrGet-NEXT:   cudaDataType value_type;
// cusparseCsrGet-NEXT:   cusparseCsrGet(
// cusparseCsrGet-NEXT:       desc /*cusparseSpMatDescr_t*/, &rows /*int64_t **/, &cols /*int64_t **/,
// cusparseCsrGet-NEXT:       &nnz /*int64_t **/, &row_ptr /*void ***/, &col_ind /*void ***/,
// cusparseCsrGet-NEXT:       &value /*void ***/, &row_ptr_type /*cusparseIndexType_t **/,
// cusparseCsrGet-NEXT:       &col_ind_type /*cusparseIndexType_t **/, &base /*cusparseIndexBase_t **/,
// cusparseCsrGet-NEXT:       &value_type /*cudaDataType **/);
// cusparseCsrGet-NEXT: Is migrated to:
// cusparseCsrGet-NEXT:   int64_t rows;
// cusparseCsrGet-NEXT:   int64_t cols;
// cusparseCsrGet-NEXT:   int64_t nnz;
// cusparseCsrGet-NEXT:   void *row_ptr;
// cusparseCsrGet-NEXT:   void *col_ind;
// cusparseCsrGet-NEXT:   void *value;
// cusparseCsrGet-NEXT:   dpct::library_data_t row_ptr_type;
// cusparseCsrGet-NEXT:   dpct::library_data_t col_ind_type;
// cusparseCsrGet-NEXT:   oneapi::mkl::index_base base;
// cusparseCsrGet-NEXT:   dpct::library_data_t value_type;
// cusparseCsrGet-NEXT:   desc->get_desc(&rows, &cols, &nnz, &row_ptr, &col_ind, &value, &row_ptr_type, &col_ind_type, &base, &value_type);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseCsrSetPointers | FileCheck %s -check-prefix=cusparseCsrSetPointers
// cusparseCsrSetPointers: CUDA API:
// cusparseCsrSetPointers-NEXT:   cusparseCsrSetPointers(desc /*cusparseSpMatDescr_t*/, row_ptr /*void **/,
// cusparseCsrSetPointers-NEXT:                          col_ind /*void **/, value /*void **/);
// cusparseCsrSetPointers-NEXT: Is migrated to:
// cusparseCsrSetPointers-NEXT:   desc->set_pointers(row_ptr, col_ind, value);
