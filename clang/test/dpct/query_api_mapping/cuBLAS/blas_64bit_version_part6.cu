// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-11.8
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6, v11.7, v11.8

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasGetVectorAsync_64 | FileCheck %s -check-prefix=cublasGetVectorAsync_64
// cublasGetVectorAsync_64: CUDA API:
// cublasGetVectorAsync_64-NEXT:   cublasGetVectorAsync_64(n /*int64_t*/, elementsize /*int64_t*/,
// cublasGetVectorAsync_64-NEXT:                           from /*const void **/, incx /*int64_t*/,
// cublasGetVectorAsync_64-NEXT:                           to /*void **/, incy /*int64_t*/,
// cublasGetVectorAsync_64-NEXT:                           stream /*cudaStream_t*/);
// cublasGetVectorAsync_64-NEXT: Is migrated to:
// cublasGetVectorAsync_64-NEXT:   dpct::blas::matrix_mem_copy(to, from, incy, incx, 1, n, elementsize, dpct::cs::memcpy_direction::automatic, *stream, true);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSetVectorAsync_64 | FileCheck %s -check-prefix=cublasSetVectorAsync_64
// cublasSetVectorAsync_64: CUDA API:
// cublasSetVectorAsync_64-NEXT:   cublasSetVectorAsync_64(n /*int64_t*/, elementsize /*int64_t*/,
// cublasSetVectorAsync_64-NEXT:                           from /*const void **/, incx /*int64_t*/,
// cublasSetVectorAsync_64-NEXT:                           to /*void **/, incy /*int64_t*/,
// cublasSetVectorAsync_64-NEXT:                           stream /*cudaStream_t*/);
// cublasSetVectorAsync_64-NEXT: Is migrated to:
// cublasSetVectorAsync_64-NEXT:   dpct::blas::matrix_mem_copy(to, from, incy, incx, 1, n, elementsize, dpct::cs::memcpy_direction::automatic, *stream, true);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasGetVector_64 | FileCheck %s -check-prefix=cublasGetVector_64
// cublasGetVector_64: CUDA API:
// cublasGetVector_64-NEXT:   cublasGetVector_64(n /*int64_t*/, elementsize /*int64_t*/, x /*const void **/,
// cublasGetVector_64-NEXT:                      incx /*int64_t*/, y /*void **/, incy /*int64_t*/);
// cublasGetVector_64-NEXT: Is migrated to:
// cublasGetVector_64-NEXT:   dpct::blas::matrix_mem_copy(y, x, incy, incx, 1, n, elementsize);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSetVector_64 | FileCheck %s -check-prefix=cublasSetVector_64
// cublasSetVector_64: CUDA API:
// cublasSetVector_64-NEXT:   cublasSetVector_64(n /*int64_t*/, elementsize /*int64_t*/, x /*const void **/,
// cublasSetVector_64-NEXT:                      incx /*int64_t*/, y /*void **/, incy /*int64_t*/);
// cublasSetVector_64-NEXT: Is migrated to:
// cublasSetVector_64-NEXT:   dpct::blas::matrix_mem_copy(y, x, incy, incx, 1, n, elementsize);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasIsamax_64 | FileCheck %s -check-prefix=cublasIsamax_64
// cublasIsamax_64: CUDA API:
// cublasIsamax_64-NEXT:   cublasIsamax_64(handle /*cublasHandle_t*/, n /*int64_t*/, x /*const float **/,
// cublasIsamax_64-NEXT:                   incx /*int64_t*/, res /*int64_t **/);
// cublasIsamax_64-NEXT: Is migrated to (with the option --no-dry-pattern):
// cublasIsamax_64-NEXT:   [&]() {
// cublasIsamax_64-NEXT:   dpct::blas::wrapper_int64_out res_wrapper_ct4(handle->get_queue(), res);
// cublasIsamax_64-NEXT:   oneapi::mkl::blas::column_major::iamax(handle->get_queue(), n, x, incx, res_wrapper_ct4.get_ptr(), oneapi::mkl::index_base::one);
// cublasIsamax_64-NEXT:   return 0;
// cublasIsamax_64-NEXT:   }();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasIdamax_64 | FileCheck %s -check-prefix=cublasIdamax_64
// cublasIdamax_64: CUDA API:
// cublasIdamax_64-NEXT:   cublasIdamax_64(handle /*cublasHandle_t*/, n /*int64_t*/,
// cublasIdamax_64-NEXT:                   x /*const double **/, incx /*int64_t*/, res /*int64_t **/);
// cublasIdamax_64-NEXT: Is migrated to (with the option --no-dry-pattern):
// cublasIdamax_64-NEXT:   [&]() {
// cublasIdamax_64-NEXT:   dpct::blas::wrapper_int64_out res_wrapper_ct4(handle->get_queue(), res);
// cublasIdamax_64-NEXT:   oneapi::mkl::blas::column_major::iamax(handle->get_queue(), n, x, incx, res_wrapper_ct4.get_ptr(), oneapi::mkl::index_base::one);
// cublasIdamax_64-NEXT:   return 0;
// cublasIdamax_64-NEXT:   }();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasIcamax_64 | FileCheck %s -check-prefix=cublasIcamax_64
// cublasIcamax_64: CUDA API:
// cublasIcamax_64-NEXT:   cublasIcamax_64(handle /*cublasHandle_t*/, n /*int64_t*/,
// cublasIcamax_64-NEXT:                   x /*const cuComplex **/, incx /*int64_t*/, res /*int64_t **/);
// cublasIcamax_64-NEXT: Is migrated to (with the option --no-dry-pattern):
// cublasIcamax_64-NEXT:   [&]() {
// cublasIcamax_64-NEXT:   dpct::blas::wrapper_int64_out res_wrapper_ct4(handle->get_queue(), res);
// cublasIcamax_64-NEXT:   oneapi::mkl::blas::column_major::iamax(handle->get_queue(), n, (std::complex<float>*)x, incx, res_wrapper_ct4.get_ptr(), oneapi::mkl::index_base::one);
// cublasIcamax_64-NEXT:   return 0;
// cublasIcamax_64-NEXT:   }();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasIzamax_64 | FileCheck %s -check-prefix=cublasIzamax_64
// cublasIzamax_64: CUDA API:
// cublasIzamax_64-NEXT:   cublasIzamax_64(handle /*cublasHandle_t*/, n /*int64_t*/,
// cublasIzamax_64-NEXT:                   x /*const cuDoubleComplex **/, incx /*int64_t*/,
// cublasIzamax_64-NEXT:                   res /*int64_t **/);
// cublasIzamax_64-NEXT: Is migrated to (with the option --no-dry-pattern):
// cublasIzamax_64-NEXT:   [&]() {
// cublasIzamax_64-NEXT:   dpct::blas::wrapper_int64_out res_wrapper_ct4(handle->get_queue(), res);
// cublasIzamax_64-NEXT:   oneapi::mkl::blas::column_major::iamax(handle->get_queue(), n, (std::complex<double>*)x, incx, res_wrapper_ct4.get_ptr(), oneapi::mkl::index_base::one);
// cublasIzamax_64-NEXT:   return 0;
// cublasIzamax_64-NEXT:   }();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasIsamin_64 | FileCheck %s -check-prefix=cublasIsamin_64
// cublasIsamin_64: CUDA API:
// cublasIsamin_64-NEXT:   cublasIsamin_64(handle /*cublasHandle_t*/, n /*int64_t*/, x /*const float **/,
// cublasIsamin_64-NEXT:                   incx /*int64_t*/, res /*int64_t **/);
// cublasIsamin_64-NEXT: Is migrated to (with the option --no-dry-pattern):
// cublasIsamin_64-NEXT:   [&]() {
// cublasIsamin_64-NEXT:   dpct::blas::wrapper_int64_out res_wrapper_ct4(handle->get_queue(), res);
// cublasIsamin_64-NEXT:   oneapi::mkl::blas::column_major::iamin(handle->get_queue(), n, x, incx, res_wrapper_ct4.get_ptr(), oneapi::mkl::index_base::one);
// cublasIsamin_64-NEXT:   return 0;
// cublasIsamin_64-NEXT:   }();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasIdamin_64 | FileCheck %s -check-prefix=cublasIdamin_64
// cublasIdamin_64: CUDA API:
// cublasIdamin_64-NEXT:   cublasIdamin_64(handle /*cublasHandle_t*/, n /*int64_t*/,
// cublasIdamin_64-NEXT:                   x /*const double **/, incx /*int64_t*/, res /*int64_t **/);
// cublasIdamin_64-NEXT: Is migrated to (with the option --no-dry-pattern):
// cublasIdamin_64-NEXT:   [&]() {
// cublasIdamin_64-NEXT:   dpct::blas::wrapper_int64_out res_wrapper_ct4(handle->get_queue(), res);
// cublasIdamin_64-NEXT:   oneapi::mkl::blas::column_major::iamin(handle->get_queue(), n, x, incx, res_wrapper_ct4.get_ptr(), oneapi::mkl::index_base::one);
// cublasIdamin_64-NEXT:   return 0;
// cublasIdamin_64-NEXT:   }();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasIcamin_64 | FileCheck %s -check-prefix=cublasIcamin_64
// cublasIcamin_64: CUDA API:
// cublasIcamin_64-NEXT:   cublasIcamin_64(handle /*cublasHandle_t*/, n /*int64_t*/,
// cublasIcamin_64-NEXT:                   x /*const cuComplex **/, incx /*int64_t*/, res /*int64_t **/);
// cublasIcamin_64-NEXT: Is migrated to (with the option --no-dry-pattern):
// cublasIcamin_64-NEXT:   [&]() {
// cublasIcamin_64-NEXT:   dpct::blas::wrapper_int64_out res_wrapper_ct4(handle->get_queue(), res);
// cublasIcamin_64-NEXT:   oneapi::mkl::blas::column_major::iamin(handle->get_queue(), n, (std::complex<float>*)x, incx, res_wrapper_ct4.get_ptr(), oneapi::mkl::index_base::one);
// cublasIcamin_64-NEXT:   return 0;
// cublasIcamin_64-NEXT:   }();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasIzamin_64 | FileCheck %s -check-prefix=cublasIzamin_64
// cublasIzamin_64: CUDA API:
// cublasIzamin_64-NEXT:   cublasIzamin_64(handle /*cublasHandle_t*/, n /*int64_t*/,
// cublasIzamin_64-NEXT:                   x /*const cuDoubleComplex **/, incx /*int64_t*/,
// cublasIzamin_64-NEXT:                   res /*int64_t **/);
// cublasIzamin_64-NEXT: Is migrated to (with the option --no-dry-pattern):
// cublasIzamin_64-NEXT:   [&]() {
// cublasIzamin_64-NEXT:   dpct::blas::wrapper_int64_out res_wrapper_ct4(handle->get_queue(), res);
// cublasIzamin_64-NEXT:   oneapi::mkl::blas::column_major::iamin(handle->get_queue(), n, (std::complex<double>*)x, incx, res_wrapper_ct4.get_ptr(), oneapi::mkl::index_base::one);
// cublasIzamin_64-NEXT:   return 0;
// cublasIzamin_64-NEXT:   }();
