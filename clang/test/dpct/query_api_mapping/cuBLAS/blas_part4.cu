// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZgerc | FileCheck %s -check-prefix=cublasZgerc
// cublasZgerc: CUDA API:
// cublasZgerc-NEXT:   cublasZgerc(handle /*cublasHandle_t*/, m /*int*/, n /*int*/,
// cublasZgerc-NEXT:               alpha /*const cuDoubleComplex **/, x /*const cuDoubleComplex **/,
// cublasZgerc-NEXT:               incx /*int*/, y /*const cuDoubleComplex **/, incy /*int*/,
// cublasZgerc-NEXT:               a /*cuDoubleComplex **/, lda /*int*/);
// cublasZgerc-NEXT: Is migrated to:
// cublasZgerc-NEXT:   oneapi::mkl::blas::column_major::gerc(*handle, m, n, dpct::get_value(alpha, *handle), (std::complex<double>*)x, incx, (std::complex<double>*)y, incy, (std::complex<double>*)a, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasStbsv | FileCheck %s -check-prefix=cublasStbsv
// cublasStbsv: CUDA API:
// cublasStbsv-NEXT:   cublasStbsv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasStbsv-NEXT:               trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasStbsv-NEXT:               n /*int*/, k /*int*/, a /*const float **/, lda /*int*/,
// cublasStbsv-NEXT:               x /*float **/, incx /*int*/);
// cublasStbsv-NEXT: Is migrated to:
// cublasStbsv-NEXT:   oneapi::mkl::blas::column_major::tbsv(*handle, upper_lower, trans, unit_nonunit, n, k, a, lda, x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasStrsm | FileCheck %s -check-prefix=cublasStrsm
// cublasStrsm: CUDA API:
// cublasStrsm-NEXT:   cublasStrsm(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasStrsm-NEXT:               upper_lower /*cublasFillMode_t*/, transa /*cublasOperation_t*/,
// cublasStrsm-NEXT:               unit_diag /*cublasDiagType_t*/, m /*int*/, n /*int*/,
// cublasStrsm-NEXT:               alpha /*const float **/, a /*const float **/, lda /*int*/,
// cublasStrsm-NEXT:               b /*float **/, ldb /*int*/);
// cublasStrsm-NEXT: Is migrated to:
// cublasStrsm-NEXT:   oneapi::mkl::blas::column_major::trsm(*handle, left_right, upper_lower, transa, unit_diag, m, n, dpct::get_value(alpha, *handle), a, lda, b, ldb);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCdotc | FileCheck %s -check-prefix=cublasCdotc
// cublasCdotc: CUDA API:
// cublasCdotc-NEXT:   cublasCdotc(handle /*cublasHandle_t*/, n /*int*/, x /*const cuComplex **/,
// cublasCdotc-NEXT:               incx /*int*/, y /*const cuComplex **/, incy /*int*/,
// cublasCdotc-NEXT:               res /*cuComplex **/);
// cublasCdotc-NEXT: Is migrated to:
// cublasCdotc-NEXT:   sycl::float2* res_temp_ptr_ct{{[0-9]+}} = res;
// cublasCdotc-NEXT:   if(sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasCdotc-NEXT:     res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<sycl::float2>(1, dpct::get_in_order_queue());
// cublasCdotc-NEXT:   }
// cublasCdotc-NEXT:   oneapi::mkl::blas::column_major::dotc(*handle, n, (std::complex<float>*)x, incx, (std::complex<float>*)y, incy, (std::complex<float>*)res_temp_ptr_ct{{[0-9]+}});
// cublasCdotc-NEXT:   if(sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasCdotc-NEXT:     handle->wait();
// cublasCdotc-NEXT:     *res = *res_temp_ptr_ct{{[0-9]+}};
// cublasCdotc-NEXT:     sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_in_order_queue());
// cublasCdotc-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSgeam | FileCheck %s -check-prefix=cublasSgeam
// cublasSgeam: CUDA API:
// cublasSgeam-NEXT:   cublasSgeam(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasSgeam-NEXT:               transb /*cublasOperation_t*/, m /*int*/, n /*int*/,
// cublasSgeam-NEXT:               alpha /*const float **/, a /*const float **/, lda /*int*/,
// cublasSgeam-NEXT:               beta /*const float **/, b /*const float **/, ldb /*int*/,
// cublasSgeam-NEXT:               c /*float **/, ldc /*int*/);
// cublasSgeam-NEXT: Is migrated to:
// cublasSgeam-NEXT:   oneapi::mkl::blas::column_major::omatadd(*handle, transa, transb, m, n, dpct::get_value(alpha, *handle), a, lda, dpct::get_value(beta, *handle), b, ldb, c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCgeru | FileCheck %s -check-prefix=cublasCgeru
// cublasCgeru: CUDA API:
// cublasCgeru-NEXT:   cublasCgeru(handle /*cublasHandle_t*/, m /*int*/, n /*int*/,
// cublasCgeru-NEXT:               alpha /*const cuComplex **/, x /*const cuComplex **/,
// cublasCgeru-NEXT:               incx /*int*/, y /*const cuComplex **/, incy /*int*/,
// cublasCgeru-NEXT:               a /*cuComplex **/, lda /*int*/);
// cublasCgeru-NEXT: Is migrated to:
// cublasCgeru-NEXT:   oneapi::mkl::blas::column_major::geru(*handle, m, n, dpct::get_value(alpha, *handle), (std::complex<float>*)x, incx, (std::complex<float>*)y, incy, (std::complex<float>*)a, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCgemm | FileCheck %s -check-prefix=cublasCgemm
// cublasCgemm: CUDA API:
// cublasCgemm-NEXT:   cublasCgemm(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasCgemm-NEXT:               transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
// cublasCgemm-NEXT:               alpha /*const cuComplex **/, a /*const cuComplex **/, lda /*int*/,
// cublasCgemm-NEXT:               b /*const cuComplex **/, ldb /*int*/, beta /*const cuComplex **/,
// cublasCgemm-NEXT:               c /*cuComplex **/, ldc /*int*/);
// cublasCgemm-NEXT: Is migrated to:
// cublasCgemm-NEXT:   oneapi::mkl::blas::column_major::gemm(*handle, transa, transb, m, n, k, dpct::get_value(alpha, *handle), (std::complex<float>*)a, lda, (std::complex<float>*)b, ldb, dpct::get_value(beta, *handle), (std::complex<float>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCgemm3m | FileCheck %s -check-prefix=cublasCgemm3m
// cublasCgemm3m: CUDA API:
// cublasCgemm3m-NEXT:   cublasCgemm3m(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasCgemm3m-NEXT:                 transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
// cublasCgemm3m-NEXT:                 alpha /*const cuComplex **/, a /*const cuComplex **/,
// cublasCgemm3m-NEXT:                 lda /*int*/, b /*const cuComplex **/, ldb /*int*/,
// cublasCgemm3m-NEXT:                 beta /*const cuComplex **/, c /*cuComplex **/, ldc /*int*/);
// cublasCgemm3m-NEXT: Is migrated to:
// cublasCgemm3m-NEXT:   oneapi::mkl::blas::column_major::gemm(*handle, transa, transb, m, n, k, dpct::get_value(alpha, *handle), (std::complex<float>*)a, lda, (std::complex<float>*)b, ldb, dpct::get_value(beta, *handle), (std::complex<float>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDtbsv | FileCheck %s -check-prefix=cublasDtbsv
// cublasDtbsv: CUDA API:
// cublasDtbsv-NEXT:   cublasDtbsv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDtbsv-NEXT:               trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasDtbsv-NEXT:               n /*int*/, k /*int*/, a /*const double **/, lda /*int*/,
// cublasDtbsv-NEXT:               x /*double **/, incx /*int*/);
// cublasDtbsv-NEXT: Is migrated to:
// cublasDtbsv-NEXT:   oneapi::mkl::blas::column_major::tbsv(*handle, upper_lower, trans, unit_nonunit, n, k, a, lda, x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZaxpy | FileCheck %s -check-prefix=cublasZaxpy
// cublasZaxpy: CUDA API:
// cublasZaxpy-NEXT:   cublasZaxpy(handle /*cublasHandle_t*/, n /*int*/,
// cublasZaxpy-NEXT:               alpha /*const cuDoubleComplex **/, x /*const cuDoubleComplex **/,
// cublasZaxpy-NEXT:               incx /*int*/, y /*cuDoubleComplex **/, incy /*int*/);
// cublasZaxpy-NEXT: Is migrated to:
// cublasZaxpy-NEXT:   oneapi::mkl::blas::column_major::axpy(*handle, n, dpct::get_value(alpha, *handle), (std::complex<double>*)x, incx, (std::complex<double>*)y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZsymv | FileCheck %s -check-prefix=cublasZsymv
// cublasZsymv: CUDA API:
// cublasZsymv-NEXT:   cublasZsymv(
// cublasZsymv-NEXT:       handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/, n /*int*/,
// cublasZsymv-NEXT:       alpha /*const cuDoubleComplex **/, a /*const cuDoubleComplex **/,
// cublasZsymv-NEXT:       lda /*int*/, x /*const cuDoubleComplex **/, incx /*int*/,
// cublasZsymv-NEXT:       beta /*const cuDoubleComplex **/, y /*cuDoubleComplex **/, incy /*int*/);
// cublasZsymv-NEXT: Is migrated to:
// cublasZsymv-NEXT:   oneapi::mkl::blas::column_major::symv(*handle, upper_lower, n, dpct::get_value(alpha, *handle), (std::complex<double>*)a, lda, (std::complex<double>*)x, incx, dpct::get_value(beta, *handle), (std::complex<double>*)y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSswap | FileCheck %s -check-prefix=cublasSswap
// cublasSswap: CUDA API:
// cublasSswap-NEXT:   cublasSswap(handle /*cublasHandle_t*/, n /*int*/, x /*float **/, incx /*int*/,
// cublasSswap-NEXT:               y /*float **/, incy /*int*/);
// cublasSswap-NEXT: Is migrated to:
// cublasSswap-NEXT:   oneapi::mkl::blas::column_major::swap(*handle, n, x, incx, y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZtrmm | FileCheck %s -check-prefix=cublasZtrmm
// cublasZtrmm: CUDA API:
// cublasZtrmm-NEXT:   cublasZtrmm(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasZtrmm-NEXT:               upper_lower /*cublasFillMode_t*/, transa /*cublasOperation_t*/,
// cublasZtrmm-NEXT:               unit_diag /*cublasDiagType_t*/, m /*int*/, n /*int*/,
// cublasZtrmm-NEXT:               alpha /*const cuDoubleComplex **/, a /*const cuDoubleComplex **/,
// cublasZtrmm-NEXT:               lda /*int*/, b /*const cuDoubleComplex **/, ldb /*int*/,
// cublasZtrmm-NEXT:               c /*cuDoubleComplex **/, ldc /*int*/);
// cublasZtrmm-NEXT: Is migrated to:
// cublasZtrmm-NEXT:   dpct::trmm(*handle, left_right, upper_lower, transa, unit_diag, m, n, alpha, a, lda, b, ldb, c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasStrmm | FileCheck %s -check-prefix=cublasStrmm
// cublasStrmm: CUDA API:
// cublasStrmm-NEXT:   cublasStrmm(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasStrmm-NEXT:               upper_lower /*cublasFillMode_t*/, transa /*cublasOperation_t*/,
// cublasStrmm-NEXT:               unit_diag /*cublasDiagType_t*/, m /*int*/, n /*int*/,
// cublasStrmm-NEXT:               alpha /*const float **/, a /*const float **/, lda /*int*/,
// cublasStrmm-NEXT:               b /*const float **/, ldb /*int*/, c /*float **/, ldc /*int*/);
// cublasStrmm-NEXT: Is migrated to:
// cublasStrmm-NEXT:   dpct::trmm(*handle, left_right, upper_lower, transa, unit_diag, m, n, alpha, a, lda, b, ldb, c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDnrm2 | FileCheck %s -check-prefix=cublasDnrm2
// cublasDnrm2: CUDA API:
// cublasDnrm2-NEXT:   cublasDnrm2(handle /*cublasHandle_t*/, n /*int*/, x /*const double **/,
// cublasDnrm2-NEXT:               incx /*int*/, res /*double **/);
// cublasDnrm2-NEXT: Is migrated to:
// cublasDnrm2-NEXT:   double* res_temp_ptr_ct{{[0-9]+}} = res;
// cublasDnrm2-NEXT:   if(sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasDnrm2-NEXT:     res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<double>(1, dpct::get_in_order_queue());
// cublasDnrm2-NEXT:   }
// cublasDnrm2-NEXT:   oneapi::mkl::blas::column_major::nrm2(*handle, n, x, incx, res_temp_ptr_ct{{[0-9]+}});
// cublasDnrm2-NEXT:   if(sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasDnrm2-NEXT:     handle->wait();
// cublasDnrm2-NEXT:     *res = *res_temp_ptr_ct{{[0-9]+}};
// cublasDnrm2-NEXT:     sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_in_order_queue());
// cublasDnrm2-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDrot | FileCheck %s -check-prefix=cublasDrot
// cublasDrot: CUDA API:
// cublasDrot-NEXT:   cublasDrot(handle /*cublasHandle_t*/, n /*int*/, x /*double **/, incx /*int*/,
// cublasDrot-NEXT:              y /*double **/, incy /*int*/, c /*const double **/,
// cublasDrot-NEXT:              s /*const double **/);
// cublasDrot-NEXT: Is migrated to:
// cublasDrot-NEXT:   oneapi::mkl::blas::column_major::rot(*handle, n, x, incx, y, incy, dpct::get_value(c, *handle), dpct::get_value(s, *handle));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDznrm2 | FileCheck %s -check-prefix=cublasDznrm2
// cublasDznrm2: CUDA API:
// cublasDznrm2-NEXT:   cublasDznrm2(handle /*cublasHandle_t*/, n /*int*/,
// cublasDznrm2-NEXT:                x /*const cuDoubleComplex **/, incx /*int*/, res /*double **/);
// cublasDznrm2-NEXT: Is migrated to:
// cublasDznrm2-NEXT:   double* res_temp_ptr_ct{{[0-9]+}} = res;
// cublasDznrm2-NEXT:   if(sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasDznrm2-NEXT:     res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<double>(1, dpct::get_in_order_queue());
// cublasDznrm2-NEXT:   }
// cublasDznrm2-NEXT:   oneapi::mkl::blas::column_major::nrm2(*handle, n, (std::complex<double>*)x, incx, res_temp_ptr_ct{{[0-9]+}});
// cublasDznrm2-NEXT:   if(sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasDznrm2-NEXT:     handle->wait();
// cublasDznrm2-NEXT:     *res = *res_temp_ptr_ct{{[0-9]+}};
// cublasDznrm2-NEXT:     sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_in_order_queue());
// cublasDznrm2-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSgbmv | FileCheck %s -check-prefix=cublasSgbmv
// cublasSgbmv: CUDA API:
// cublasSgbmv-NEXT:   cublasSgbmv(handle /*cublasHandle_t*/, trans /*cublasOperation_t*/, m /*int*/,
// cublasSgbmv-NEXT:               n /*int*/, kl /*int*/, ku /*int*/, alpha /*const float **/,
// cublasSgbmv-NEXT:               a /*const float **/, lda /*int*/, x /*const float **/,
// cublasSgbmv-NEXT:               incx /*int*/, beta /*const float **/, y /*float **/,
// cublasSgbmv-NEXT:               incy /*int*/);
// cublasSgbmv-NEXT: Is migrated to:
// cublasSgbmv-NEXT:   oneapi::mkl::blas::column_major::gbmv(*handle, trans, m, n, kl, ku, dpct::get_value(alpha, *handle), a, lda, x, incx, dpct::get_value(beta, *handle), y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSetVectorAsync | FileCheck %s -check-prefix=cublasSetVectorAsync
// cublasSetVectorAsync: CUDA API:
// cublasSetVectorAsync-NEXT:   cublasSetVectorAsync(n /*int*/, elementsize /*int*/, from /*const void **/,
// cublasSetVectorAsync-NEXT:                        incx /*int*/, to /*void **/, incy /*int*/,
// cublasSetVectorAsync-NEXT:                        stream /*cudaStream_t*/);
// cublasSetVectorAsync-NEXT: Is migrated to:
// cublasSetVectorAsync-NEXT:   dpct::matrix_mem_copy((void*)to, (void*)from, incy, incx, 1, n, elementsize, dpct::automatic, *stream, true);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSsymv | FileCheck %s -check-prefix=cublasSsymv
// cublasSsymv: CUDA API:
// cublasSsymv-NEXT:   cublasSsymv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasSsymv-NEXT:               n /*int*/, alpha /*const float **/, a /*const float **/,
// cublasSsymv-NEXT:               lda /*int*/, x /*const float **/, incx /*int*/,
// cublasSsymv-NEXT:               beta /*const float **/, y /*float **/, incy /*int*/);
// cublasSsymv-NEXT: Is migrated to:
// cublasSsymv-NEXT:   oneapi::mkl::blas::column_major::symv(*handle, upper_lower, n, dpct::get_value(alpha, *handle), a, lda, x, incx, dpct::get_value(beta, *handle), y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCtrsv | FileCheck %s -check-prefix=cublasCtrsv
// cublasCtrsv: CUDA API:
// cublasCtrsv-NEXT:   cublasCtrsv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCtrsv-NEXT:               trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasCtrsv-NEXT:               n /*int*/, a /*const cuComplex **/, lda /*int*/,
// cublasCtrsv-NEXT:               x /*cuComplex **/, incx /*int*/);
// cublasCtrsv-NEXT: Is migrated to:
// cublasCtrsv-NEXT:   oneapi::mkl::blas::column_major::trsv(*handle, upper_lower, trans, unit_nonunit, n, (std::complex<float>*)a, lda, (std::complex<float>*)x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDasum | FileCheck %s -check-prefix=cublasDasum
// cublasDasum: CUDA API:
// cublasDasum-NEXT:   cublasDasum(handle /*cublasHandle_t*/, n /*int*/, x /*const double **/,
// cublasDasum-NEXT:               incx /*int*/, res /*double **/);
// cublasDasum-NEXT: Is migrated to:
// cublasDasum-NEXT:   double* res_temp_ptr_ct{{[0-9]+}} = res;
// cublasDasum-NEXT:   if(sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasDasum-NEXT:     res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<double>(1, dpct::get_in_order_queue());
// cublasDasum-NEXT:   }
// cublasDasum-NEXT:   oneapi::mkl::blas::column_major::asum(*handle, n, x, incx, res_temp_ptr_ct{{[0-9]+}});
// cublasDasum-NEXT:   if(sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasDasum-NEXT:     handle->wait();
// cublasDasum-NEXT:     *res = *res_temp_ptr_ct{{[0-9]+}};
// cublasDasum-NEXT:     sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_in_order_queue());
// cublasDasum-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSsbmv | FileCheck %s -check-prefix=cublasSsbmv
// cublasSsbmv: CUDA API:
// cublasSsbmv-NEXT:   cublasSsbmv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasSsbmv-NEXT:               n /*int*/, k /*int*/, alpha /*const float **/,
// cublasSsbmv-NEXT:               a /*const float **/, lda /*int*/, x /*const float **/,
// cublasSsbmv-NEXT:               incx /*int*/, beta /*const float **/, y /*float **/,
// cublasSsbmv-NEXT:               incy /*int*/);
// cublasSsbmv-NEXT: Is migrated to:
// cublasSsbmv-NEXT:   oneapi::mkl::blas::column_major::sbmv(*handle, upper_lower, n, k, dpct::get_value(alpha, *handle), a, lda, x, incx, dpct::get_value(beta, *handle), y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasIzamin | FileCheck %s -check-prefix=cublasIzamin
// cublasIzamin: CUDA API:
// cublasIzamin-NEXT:   cublasIzamin(handle /*cublasHandle_t*/, n /*int*/,
// cublasIzamin-NEXT:                x /*const cuDoubleComplex **/, incx /*int*/, res /*int **/);
// cublasIzamin-NEXT: Is migrated to (with the option --no-dry-pattern):
// cublasIzamin-NEXT:   int64_t* res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<int64_t>(1, dpct::get_in_order_queue());
// cublasIzamin-NEXT:   oneapi::mkl::blas::column_major::iamin(*handle, n, (std::complex<double>*)x, incx, res_temp_ptr_ct{{[0-9]+}}, oneapi::mkl::index_base::one).wait();
// cublasIzamin-NEXT:   int res_temp_host_ct{{[0-9]+}} = (int)*res_temp_ptr_ct{{[0-9]+}};
// cublasIzamin-NEXT:   dpct::dpct_memcpy(res, &res_temp_host_ct{{[0-9]+}}, sizeof(int));
// cublasIzamin-NEXT:   sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_in_order_queue());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasNrm2Ex | FileCheck %s -check-prefix=cublasNrm2Ex
// cublasNrm2Ex: CUDA API:
// cublasNrm2Ex-NEXT:   cublasNrm2Ex(handle /*cublasHandle_t*/, n /*int*/, x /*const void **/,
// cublasNrm2Ex-NEXT:                xtype /*cudaDataType*/, incx /*int*/, res /*void **/,
// cublasNrm2Ex-NEXT:                restype /*cudaDataType*/, computetype /*cudaDataType*/);
// cublasNrm2Ex-NEXT: Is migrated to:
// cublasNrm2Ex-NEXT:   dpct::nrm2(*handle, n, x, xtype, incx, res, restype);
