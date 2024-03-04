// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCsscal | FileCheck %s -check-prefix=cublasCsscal
// cublasCsscal: CUDA API:
// cublasCsscal-NEXT:   cublasCsscal(handle /*cublasHandle_t*/, n /*int*/, alpha /*const float **/,
// cublasCsscal-NEXT:                x /*cuComplex **/, incx /*int*/);
// cublasCsscal-NEXT: Is migrated to:
// cublasCsscal-NEXT:   oneapi::mkl::blas::column_major::scal(handle->get_queue(), n, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasIdamin | FileCheck %s -check-prefix=cublasIdamin
// cublasIdamin: CUDA API:
// cublasIdamin-NEXT:   cublasIdamin(handle /*cublasHandle_t*/, n /*int*/, x /*const double **/,
// cublasIdamin-NEXT:                incx /*int*/, res /*int **/);
// cublasIdamin-NEXT: Is migrated to (with the option --no-dry-pattern):
// cublasIdamin-NEXT:   [&]() {
// cublasIdamin-NEXT:   dpct::blas::wrapper_int_to_int64_out res_wrapper_ct4(handle->get_queue(), res);
// cublasIdamin-NEXT:   oneapi::mkl::blas::column_major::iamin(handle->get_queue(), n, x, incx, res_wrapper_ct4.get_ptr(), oneapi::mkl::index_base::one);
// cublasIdamin-NEXT:   return 0;
// cublasIdamin-NEXT:   }();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSetMatrix | FileCheck %s -check-prefix=cublasSetMatrix
// cublasSetMatrix: CUDA API:
// cublasSetMatrix-NEXT:   cublasSetMatrix(rows /*int*/, cols /*int*/, elementsize /*int*/,
// cublasSetMatrix-NEXT:                   a /*const void **/, lda /*int*/, b /*void **/, ldb /*int*/);
// cublasSetMatrix-NEXT: Is migrated to:
// cublasSetMatrix-NEXT:   dpct::matrix_mem_copy((void*)b, (void*)a, ldb, lda, rows, cols, elementsize);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZscal | FileCheck %s -check-prefix=cublasZscal
// cublasZscal: CUDA API:
// cublasZscal-NEXT:   cublasZscal(handle /*cublasHandle_t*/, n /*int*/,
// cublasZscal-NEXT:               alpha /*const cuDoubleComplex **/, x /*cuDoubleComplex **/,
// cublasZscal-NEXT:               incx /*int*/);
// cublasZscal-NEXT: Is migrated to:
// cublasZscal-NEXT:   oneapi::mkl::blas::column_major::scal(handle->get_queue(), n, dpct::get_value(alpha, handle->get_queue()), (std::complex<double>*)x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCtrmv | FileCheck %s -check-prefix=cublasCtrmv
// cublasCtrmv: CUDA API:
// cublasCtrmv-NEXT:   cublasCtrmv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCtrmv-NEXT:               trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasCtrmv-NEXT:               n /*int*/, a /*const cuComplex **/, lda /*int*/,
// cublasCtrmv-NEXT:               x /*cuComplex **/, incx /*int*/);
// cublasCtrmv-NEXT: Is migrated to:
// cublasCtrmv-NEXT:   oneapi::mkl::blas::column_major::trmv(handle->get_queue(), upper_lower, trans, unit_nonunit, n, (std::complex<float>*)a, lda, (std::complex<float>*)x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZrot | FileCheck %s -check-prefix=cublasZrot
// cublasZrot: CUDA API:
// cublasZrot-NEXT:   cublasZrot(handle /*cublasHandle_t*/, n /*int*/, x /*cuDoubleComplex **/,
// cublasZrot-NEXT:              incx /*int*/, y /*cuDoubleComplex **/, incy /*int*/,
// cublasZrot-NEXT:              c /*const double **/, s /*const cuDoubleComplex **/);
// cublasZrot-NEXT: Is migrated to:
// cublasZrot-NEXT:   dpct::rot(handle->get_queue(), n, x, dpct::library_data_t::complex_double, incx, y, dpct::library_data_t::complex_double, incy, c, s, dpct::library_data_t::complex_double);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSetVector | FileCheck %s -check-prefix=cublasSetVector
// cublasSetVector: CUDA API:
// cublasSetVector-NEXT:   cublasSetVector(n /*int*/, elementsize /*int*/, x /*const void **/,
// cublasSetVector-NEXT:                   incx /*int*/, y /*void **/, incy /*int*/);
// cublasSetVector-NEXT: Is migrated to:
// cublasSetVector-NEXT:   dpct::matrix_mem_copy((void*)y, (void*)x, incy, incx, 1, n, elementsize);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSetStream | FileCheck %s -check-prefix=cublasSetStream
// cublasSetStream: CUDA API:
// cublasSetStream-NEXT:   cublasSetStream(handle /*cublasHandle_t*/, stream /*cudaStream_t*/);
// cublasSetStream-NEXT: Is migrated to:
// cublasSetStream-NEXT:   handle->set_queue(stream);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZdgmm | FileCheck %s -check-prefix=cublasZdgmm
// cublasZdgmm: CUDA API:
// cublasZdgmm-NEXT:   cublasZdgmm(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasZdgmm-NEXT:               m /*int*/, n /*int*/, a /*const cuDoubleComplex **/, lda /*int*/,
// cublasZdgmm-NEXT:               x /*const cuDoubleComplex **/, incx /*int*/,
// cublasZdgmm-NEXT:               c /*cuDoubleComplex **/, ldc /*int*/);
// cublasZdgmm-NEXT: Is migrated to:
// cublasZdgmm-NEXT:   oneapi::mkl::blas::column_major::dgmm(handle->get_queue(), left_right, m, n, (std::complex<double>*)a, lda, (std::complex<double>*)x, incx, (std::complex<double>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZdrot | FileCheck %s -check-prefix=cublasZdrot
// cublasZdrot: CUDA API:
// cublasZdrot-NEXT:   cublasZdrot(handle /*cublasHandle_t*/, n /*int*/, x /*cuDoubleComplex **/,
// cublasZdrot-NEXT:               incx /*int*/, y /*cuDoubleComplex **/, incy /*int*/,
// cublasZdrot-NEXT:               c /*const double **/, s /*const double **/);
// cublasZdrot-NEXT: Is migrated to:
// cublasZdrot-NEXT:   oneapi::mkl::blas::column_major::rot(handle->get_queue(), n, (std::complex<double>*)x, incx, (std::complex<double>*)y, incy, dpct::get_value(c, handle->get_queue()), dpct::get_value(s, handle->get_queue()));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZhpr | FileCheck %s -check-prefix=cublasZhpr
// cublasZhpr: CUDA API:
// cublasZhpr-NEXT:   cublasZhpr(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZhpr-NEXT:              n /*int*/, alpha /*const double **/, x /*const cuDoubleComplex **/,
// cublasZhpr-NEXT:              incx /*int*/, a /*cuDoubleComplex **/);
// cublasZhpr-NEXT: Is migrated to:
// cublasZhpr-NEXT:   oneapi::mkl::blas::column_major::hpr(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<double>*)x, incx, (std::complex<double>*)a);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCtpmv | FileCheck %s -check-prefix=cublasCtpmv
// cublasCtpmv: CUDA API:
// cublasCtpmv-NEXT:   cublasCtpmv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCtpmv-NEXT:               trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasCtpmv-NEXT:               n /*int*/, a /*const cuComplex **/, x /*cuComplex **/,
// cublasCtpmv-NEXT:               incx /*int*/);
// cublasCtpmv-NEXT: Is migrated to:
// cublasCtpmv-NEXT:   oneapi::mkl::blas::column_major::tpmv(handle->get_queue(), upper_lower, trans, unit_nonunit, n, (std::complex<float>*)a, (std::complex<float>*)x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDtrmv | FileCheck %s -check-prefix=cublasDtrmv
// cublasDtrmv: CUDA API:
// cublasDtrmv-NEXT:   cublasDtrmv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDtrmv-NEXT:               trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasDtrmv-NEXT:               n /*int*/, a /*const double **/, lda /*int*/, x /*double **/,
// cublasDtrmv-NEXT:               incx /*int*/);
// cublasDtrmv-NEXT: Is migrated to:
// cublasDtrmv-NEXT:   oneapi::mkl::blas::column_major::trmv(handle->get_queue(), upper_lower, trans, unit_nonunit, n, a, lda, x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCtrsm | FileCheck %s -check-prefix=cublasCtrsm
// cublasCtrsm: CUDA API:
// cublasCtrsm-NEXT:   cublasCtrsm(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasCtrsm-NEXT:               upper_lower /*cublasFillMode_t*/, transa /*cublasOperation_t*/,
// cublasCtrsm-NEXT:               unit_diag /*cublasDiagType_t*/, m /*int*/, n /*int*/,
// cublasCtrsm-NEXT:               alpha /*const cuComplex **/, a /*const cuComplex **/, lda /*int*/,
// cublasCtrsm-NEXT:               b /*cuComplex **/, ldb /*int*/);
// cublasCtrsm-NEXT: Is migrated to:
// cublasCtrsm-NEXT:   oneapi::mkl::blas::column_major::trsm(handle->get_queue(), left_right, upper_lower, transa, unit_diag, m, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)a, lda, (std::complex<float>*)b, ldb);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCsyrk | FileCheck %s -check-prefix=cublasCsyrk
// cublasCsyrk: CUDA API:
// cublasCsyrk-NEXT:   cublasCsyrk(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCsyrk-NEXT:               trans /*cublasOperation_t*/, n /*int*/, k /*int*/,
// cublasCsyrk-NEXT:               alpha /*const cuComplex **/, a /*const cuComplex **/, lda /*int*/,
// cublasCsyrk-NEXT:               beta /*const cuComplex **/, c /*cuComplex **/, ldc /*int*/);
// cublasCsyrk-NEXT: Is migrated to:
// cublasCsyrk-NEXT:   oneapi::mkl::blas::column_major::syrk(handle->get_queue(), upper_lower, trans, n, k, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)a, lda, dpct::get_value(beta, handle->get_queue()), (std::complex<float>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDsyr2 | FileCheck %s -check-prefix=cublasDsyr2
// cublasDsyr2: CUDA API:
// cublasDsyr2-NEXT:   cublasDsyr2(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDsyr2-NEXT:               n /*int*/, alpha /*const double **/, x /*const double **/,
// cublasDsyr2-NEXT:               incx /*int*/, y /*const double **/, incy /*int*/, a /*double **/,
// cublasDsyr2-NEXT:               lda /*int*/);
// cublasDsyr2-NEXT: Is migrated to:
// cublasDsyr2-NEXT:   oneapi::mkl::blas::column_major::syr2(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), x, incx, y, incy, a, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasIdamax | FileCheck %s -check-prefix=cublasIdamax
// cublasIdamax: CUDA API:
// cublasIdamax-NEXT:   cublasIdamax(handle /*cublasHandle_t*/, n /*int*/, x /*const double **/,
// cublasIdamax-NEXT:                incx /*int*/, res /*int **/);
// cublasIdamax-NEXT: Is migrated to (with the option --no-dry-pattern):
// cublasIdamax-NEXT:   [&]() {
// cublasIdamax-NEXT:   dpct::blas::wrapper_int_to_int64_out res_wrapper_ct4(handle->get_queue(), res);
// cublasIdamax-NEXT:   oneapi::mkl::blas::column_major::iamax(handle->get_queue(), n, x, incx, res_wrapper_ct4.get_ptr(), oneapi::mkl::index_base::one);
// cublasIdamax-NEXT:   return 0;
// cublasIdamax-NEXT:   }();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZgemv | FileCheck %s -check-prefix=cublasZgemv
// cublasZgemv: CUDA API:
// cublasZgemv-NEXT:   cublasZgemv(handle /*cublasHandle_t*/, trans /*cublasOperation_t*/, m /*int*/,
// cublasZgemv-NEXT:               n /*int*/, alpha /*const cuDoubleComplex **/,
// cublasZgemv-NEXT:               a /*const cuDoubleComplex **/, lda /*int*/,
// cublasZgemv-NEXT:               x /*const cuDoubleComplex **/, incx /*int*/,
// cublasZgemv-NEXT:               beta /*const cuDoubleComplex **/, y /*cuDoubleComplex **/,
// cublasZgemv-NEXT:               incy /*int*/);
// cublasZgemv-NEXT: Is migrated to:
// cublasZgemv-NEXT:   oneapi::mkl::blas::column_major::gemv(handle->get_queue(), trans, m, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<double>*)a, lda, (std::complex<double>*)x, incx, dpct::get_value(beta, handle->get_queue()), (std::complex<double>*)y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasIsamin | FileCheck %s -check-prefix=cublasIsamin
// cublasIsamin: CUDA API:
// cublasIsamin-NEXT:   cublasIsamin(handle /*cublasHandle_t*/, n /*int*/, x /*const float **/,
// cublasIsamin-NEXT:                incx /*int*/, res /*int **/);
// cublasIsamin-NEXT: Is migrated to (with the option --no-dry-pattern):
// cublasIsamin-NEXT:   [&]() {
// cublasIsamin-NEXT:   dpct::blas::wrapper_int_to_int64_out res_wrapper_ct4(handle->get_queue(), res);
// cublasIsamin-NEXT:   oneapi::mkl::blas::column_major::iamin(handle->get_queue(), n, x, incx, res_wrapper_ct4.get_ptr(), oneapi::mkl::index_base::one);
// cublasIsamin-NEXT:   return 0;
// cublasIsamin-NEXT:   }();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZswap | FileCheck %s -check-prefix=cublasZswap
// cublasZswap: CUDA API:
// cublasZswap-NEXT:   cublasZswap(handle /*cublasHandle_t*/, n /*int*/, x /*cuDoubleComplex **/,
// cublasZswap-NEXT:               incx /*int*/, y /*cuDoubleComplex **/, incy /*int*/);
// cublasZswap-NEXT: Is migrated to:
// cublasZswap-NEXT:   oneapi::mkl::blas::column_major::swap(handle->get_queue(), n, (std::complex<double>*)x, incx, (std::complex<double>*)y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSspr | FileCheck %s -check-prefix=cublasSspr
// cublasSspr: CUDA API:
// cublasSspr-NEXT:   cublasSspr(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasSspr-NEXT:              n /*int*/, alpha /*const float **/, x /*const float **/,
// cublasSspr-NEXT:              incx /*int*/, a /*float **/);
// cublasSspr-NEXT: Is migrated to:
// cublasSspr-NEXT:   oneapi::mkl::blas::column_major::spr(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), x, incx, a);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDdot | FileCheck %s -check-prefix=cublasDdot
// cublasDdot: CUDA API:
// cublasDdot-NEXT:   cublasDdot(handle /*cublasHandle_t*/, n /*int*/, x /*const double **/,
// cublasDdot-NEXT:              incx /*int*/, y /*const double **/, incy /*int*/,
// cublasDdot-NEXT:              res /*double **/);
// cublasDdot-NEXT: Is migrated to:
// cublasDdot-NEXT:   double* res_temp_ptr_ct{{[0-9]+}} = res;
// cublasDdot-NEXT:   if(sycl::get_pointer_type(res, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(res, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
// cublasDdot-NEXT:     res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<double>(1, dpct::get_in_order_queue());
// cublasDdot-NEXT:   }
// cublasDdot-NEXT:   oneapi::mkl::blas::column_major::dot(handle->get_queue(), n, x, incx, y, incy, res_temp_ptr_ct{{[0-9]+}});
// cublasDdot-NEXT:   if(sycl::get_pointer_type(res, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(res, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
// cublasDdot-NEXT:     handle->get_queue().wait();
// cublasDdot-NEXT:     *res = *res_temp_ptr_ct{{[0-9]+}};
// cublasDdot-NEXT:     sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_in_order_queue());
// cublasDdot-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCsymv | FileCheck %s -check-prefix=cublasCsymv
// cublasCsymv: CUDA API:
// cublasCsymv-NEXT:   cublasCsymv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCsymv-NEXT:               n /*int*/, alpha /*const cuComplex **/, a /*const cuComplex **/,
// cublasCsymv-NEXT:               lda /*int*/, x /*const cuComplex **/, incx /*int*/,
// cublasCsymv-NEXT:               beta /*const cuComplex **/, y /*cuComplex **/, incy /*int*/);
// cublasCsymv-NEXT: Is migrated to:
// cublasCsymv-NEXT:   oneapi::mkl::blas::column_major::symv(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)a, lda, (std::complex<float>*)x, incx, dpct::get_value(beta, handle->get_queue()), (std::complex<float>*)y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDspr2 | FileCheck %s -check-prefix=cublasDspr2
// cublasDspr2: CUDA API:
// cublasDspr2-NEXT:   cublasDspr2(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDspr2-NEXT:               n /*int*/, alpha /*const double **/, x /*const double **/,
// cublasDspr2-NEXT:               incx /*int*/, y /*const double **/, incy /*int*/, a /*double **/);
// cublasDspr2-NEXT: Is migrated to:
// cublasDspr2-NEXT:   oneapi::mkl::blas::column_major::spr2(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), x, incx, y, incy, a);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZherk | FileCheck %s -check-prefix=cublasZherk
// cublasZherk: CUDA API:
// cublasZherk-NEXT:   cublasZherk(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZherk-NEXT:               trans /*cublasOperation_t*/, n /*int*/, k /*int*/,
// cublasZherk-NEXT:               alpha /*const double **/, a /*const cuDoubleComplex **/,
// cublasZherk-NEXT:               lda /*int*/, beta /*const double **/, c /*cuDoubleComplex **/,
// cublasZherk-NEXT:               ldc /*int*/);
// cublasZherk-NEXT: Is migrated to:
// cublasZherk-NEXT:   oneapi::mkl::blas::column_major::herk(handle->get_queue(), upper_lower, trans, n, k, dpct::get_value(alpha, handle->get_queue()), (std::complex<double>*)a, lda, dpct::get_value(beta, handle->get_queue()), (std::complex<double>*)c, ldc);
