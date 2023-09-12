// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSgemmStridedBatched | FileCheck %s -check-prefix=cublasSgemmStridedBatched
// cublasSgemmStridedBatched: CUDA API:
// cublasSgemmStridedBatched-NEXT:   cublasSgemmStridedBatched(
// cublasSgemmStridedBatched-NEXT:       handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasSgemmStridedBatched-NEXT:       transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
// cublasSgemmStridedBatched-NEXT:       alpha /*const float **/, a /*const float **/, lda /*int*/,
// cublasSgemmStridedBatched-NEXT:       stridea /*long long int*/, b /*const float **/, ldb /*int*/,
// cublasSgemmStridedBatched-NEXT:       strideb /*long long int*/, beta /*const float **/, c /*float **/,
// cublasSgemmStridedBatched-NEXT:       ldc /*int*/, stridec /*long long int*/, group_count /*int*/);
// cublasSgemmStridedBatched-NEXT: Is migrated to:
// cublasSgemmStridedBatched-NEXT:   oneapi::mkl::blas::column_major::gemm_batch(*handle, transa, transb, m, n, k, dpct::get_value(alpha, *handle), a, lda, stridea, b, ldb, strideb, dpct::get_value(beta, *handle), c, ldc, stridec, group_count);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasScalEx | FileCheck %s -check-prefix=cublasScalEx
// cublasScalEx: CUDA API:
// cublasScalEx-NEXT:   cublasScalEx(handle /*cublasHandle_t*/, n /*int*/, alpha /*const void **/,
// cublasScalEx-NEXT:                alphatype /*cudaDataType*/, x /*void **/, xtype /*cudaDataType*/,
// cublasScalEx-NEXT:                incx /*int*/, computetype /*cudaDataType*/);
// cublasScalEx-NEXT: Is migrated to:
// cublasScalEx-NEXT:   dpct::scal(*handle, n, alpha, alphatype, x, xtype, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDger | FileCheck %s -check-prefix=cublasDger
// cublasDger: CUDA API:
// cublasDger-NEXT:   cublasDger(handle /*cublasHandle_t*/, m /*int*/, n /*int*/,
// cublasDger-NEXT:              alpha /*const double **/, x /*const double **/, incx /*int*/,
// cublasDger-NEXT:              y /*const double **/, incy /*int*/, a /*double **/, lda /*int*/);
// cublasDger-NEXT: Is migrated to:
// cublasDger-NEXT:   oneapi::mkl::blas::column_major::ger(*handle, m, n, dpct::get_value(alpha, *handle), x, incx, y, incy, a, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasScasum | FileCheck %s -check-prefix=cublasScasum
// cublasScasum: CUDA API:
// cublasScasum-NEXT:   cublasScasum(handle /*cublasHandle_t*/, n /*int*/, x /*const cuComplex **/,
// cublasScasum-NEXT:                incx /*int*/, res /*float **/);
// cublasScasum-NEXT: Is migrated to:
// cublasScasum-NEXT:   float* res_temp_ptr_ct{{[0-9]+}} = res;
// cublasScasum-NEXT:   if(sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasScasum-NEXT:     res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<float>(1, dpct::get_in_order_queue());
// cublasScasum-NEXT:   }
// cublasScasum-NEXT:   oneapi::mkl::blas::column_major::asum(*handle, n, (std::complex<float>*)x, incx, res_temp_ptr_ct{{[0-9]+}});
// cublasScasum-NEXT:   if(sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasScasum-NEXT:     handle->wait();
// cublasScasum-NEXT:     *res = *res_temp_ptr_ct{{[0-9]+}};
// cublasScasum-NEXT:     sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_in_order_queue());
// cublasScasum-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSsyr2k | FileCheck %s -check-prefix=cublasSsyr2k
// cublasSsyr2k: CUDA API:
// cublasSsyr2k-NEXT:   cublasSsyr2k(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasSsyr2k-NEXT:                trans /*cublasOperation_t*/, n /*int*/, k /*int*/,
// cublasSsyr2k-NEXT:                alpha /*const float **/, a /*const float **/, lda /*int*/,
// cublasSsyr2k-NEXT:                b /*const float **/, ldb /*int*/, beta /*const float **/,
// cublasSsyr2k-NEXT:                c /*float **/, ldc /*int*/);
// cublasSsyr2k-NEXT: Is migrated to:
// cublasSsyr2k-NEXT:   oneapi::mkl::blas::column_major::syr2k(*handle, upper_lower, trans, n, k, dpct::get_value(alpha, *handle), a, lda, b, ldb, dpct::get_value(beta, *handle), c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSsyr | FileCheck %s -check-prefix=cublasSsyr
// cublasSsyr: CUDA API:
// cublasSsyr-NEXT:   cublasSsyr(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasSsyr-NEXT:              n /*int*/, alpha /*const float **/, x /*const float **/,
// cublasSsyr-NEXT:              incx /*int*/, a /*float **/, lda /*int*/);
// cublasSsyr-NEXT: Is migrated to:
// cublasSsyr-NEXT:   oneapi::mkl::blas::column_major::syr(*handle, upper_lower, n, dpct::get_value(alpha, *handle), x, incx, a, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCsymm | FileCheck %s -check-prefix=cublasCsymm
// cublasCsymm: CUDA API:
// cublasCsymm-NEXT:   cublasCsymm(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasCsymm-NEXT:               upper_lower /*cublasFillMode_t*/, m /*int*/, n /*int*/,
// cublasCsymm-NEXT:               alpha /*const cuComplex **/, a /*const cuComplex **/, lda /*int*/,
// cublasCsymm-NEXT:               b /*const cuComplex **/, ldb /*int*/, beta /*const cuComplex **/,
// cublasCsymm-NEXT:               c /*cuComplex **/, ldc /*int*/);
// cublasCsymm-NEXT: Is migrated to:
// cublasCsymm-NEXT:   oneapi::mkl::blas::column_major::symm(*handle, left_right, upper_lower, m, n, dpct::get_value(alpha, *handle), (std::complex<float>*)a, lda, (std::complex<float>*)b, ldb, dpct::get_value(beta, *handle), (std::complex<float>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSger | FileCheck %s -check-prefix=cublasSger
// cublasSger: CUDA API:
// cublasSger-NEXT:   cublasSger(handle /*cublasHandle_t*/, m /*int*/, n /*int*/,
// cublasSger-NEXT:              alpha /*const float **/, x /*const float **/, incx /*int*/,
// cublasSger-NEXT:              y /*const float **/, incy /*int*/, a /*float **/, lda /*int*/);
// cublasSger-NEXT: Is migrated to:
// cublasSger-NEXT:   oneapi::mkl::blas::column_major::ger(*handle, m, n, dpct::get_value(alpha, *handle), x, incx, y, incy, a, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZdotu | FileCheck %s -check-prefix=cublasZdotu
// cublasZdotu: CUDA API:
// cublasZdotu-NEXT:   cublasZdotu(handle /*cublasHandle_t*/, n /*int*/,
// cublasZdotu-NEXT:               x /*const cuDoubleComplex **/, incx /*int*/,
// cublasZdotu-NEXT:               y /*const cuDoubleComplex **/, incy /*int*/,
// cublasZdotu-NEXT:               res /*cuDoubleComplex **/);
// cublasZdotu-NEXT: Is migrated to:
// cublasZdotu-NEXT:   sycl::double2* res_temp_ptr_ct{{[0-9]+}} = res;
// cublasZdotu-NEXT:   if(sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasZdotu-NEXT:     res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<sycl::double2>(1, dpct::get_in_order_queue());
// cublasZdotu-NEXT:   }
// cublasZdotu-NEXT:   oneapi::mkl::blas::column_major::dotu(*handle, n, (std::complex<double>*)x, incx, (std::complex<double>*)y, incy, (std::complex<double>*)res_temp_ptr_ct{{[0-9]+}});
// cublasZdotu-NEXT:   if(sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasZdotu-NEXT:     handle->wait();
// cublasZdotu-NEXT:     *res = *res_temp_ptr_ct{{[0-9]+}};
// cublasZdotu-NEXT:     sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_in_order_queue());
// cublasZdotu-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZdotc | FileCheck %s -check-prefix=cublasZdotc
// cublasZdotc: CUDA API:
// cublasZdotc-NEXT:   cublasZdotc(handle /*cublasHandle_t*/, n /*int*/,
// cublasZdotc-NEXT:               x /*const cuDoubleComplex **/, incx /*int*/,
// cublasZdotc-NEXT:               y /*const cuDoubleComplex **/, incy /*int*/,
// cublasZdotc-NEXT:               res /*cuDoubleComplex **/);
// cublasZdotc-NEXT: Is migrated to:
// cublasZdotc-NEXT:   sycl::double2* res_temp_ptr_ct{{[0-9]+}} = res;
// cublasZdotc-NEXT:   if(sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasZdotc-NEXT:     res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<sycl::double2>(1, dpct::get_in_order_queue());
// cublasZdotc-NEXT:   }
// cublasZdotc-NEXT:   oneapi::mkl::blas::column_major::dotc(*handle, n, (std::complex<double>*)x, incx, (std::complex<double>*)y, incy, (std::complex<double>*)res_temp_ptr_ct{{[0-9]+}});
// cublasZdotc-NEXT:   if(sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasZdotc-NEXT:     handle->wait();
// cublasZdotc-NEXT:     *res = *res_temp_ptr_ct{{[0-9]+}};
// cublasZdotc-NEXT:     sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_in_order_queue());
// cublasZdotc-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSsymm | FileCheck %s -check-prefix=cublasSsymm
// cublasSsymm: CUDA API:
// cublasSsymm-NEXT:   cublasSsymm(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasSsymm-NEXT:               upper_lower /*cublasFillMode_t*/, m /*int*/, n /*int*/,
// cublasSsymm-NEXT:               alpha /*const float **/, a /*const float **/, lda /*int*/,
// cublasSsymm-NEXT:               b /*const float **/, ldb /*int*/, beta /*const float **/,
// cublasSsymm-NEXT:               c /*float **/, ldc /*int*/);
// cublasSsymm-NEXT: Is migrated to:
// cublasSsymm-NEXT:   oneapi::mkl::blas::column_major::symm(*handle, left_right, upper_lower, m, n, dpct::get_value(alpha, *handle), a, lda, b, ldb, dpct::get_value(beta, *handle), c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCherk | FileCheck %s -check-prefix=cublasCherk
// cublasCherk: CUDA API:
// cublasCherk-NEXT:   cublasCherk(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCherk-NEXT:               trans /*cublasOperation_t*/, n /*int*/, k /*int*/,
// cublasCherk-NEXT:               alpha /*const float **/, a /*const cuComplex **/, lda /*int*/,
// cublasCherk-NEXT:               beta /*const float **/, c /*cuComplex **/, ldc /*int*/);
// cublasCherk-NEXT: Is migrated to:
// cublasCherk-NEXT:   oneapi::mkl::blas::column_major::herk(*handle, upper_lower, trans, n, k, dpct::get_value(alpha, *handle), (std::complex<float>*)a, lda, dpct::get_value(beta, *handle), (std::complex<float>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZhemv | FileCheck %s -check-prefix=cublasZhemv
// cublasZhemv: CUDA API:
// cublasZhemv-NEXT:   cublasZhemv(
// cublasZhemv-NEXT:       handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/, n /*int*/,
// cublasZhemv-NEXT:       alpha /*const cuDoubleComplex **/, a /*const cuDoubleComplex **/,
// cublasZhemv-NEXT:       lda /*int*/, x /*const cuDoubleComplex **/, incx /*int*/,
// cublasZhemv-NEXT:       beta /*const cuDoubleComplex **/, y /*cuDoubleComplex **/, incy /*int*/);
// cublasZhemv-NEXT: Is migrated to:
// cublasZhemv-NEXT:   oneapi::mkl::blas::column_major::hemv(*handle, upper_lower, n, dpct::get_value(alpha, *handle), (std::complex<double>*)a, lda, (std::complex<double>*)x, incx, dpct::get_value(beta, *handle), (std::complex<double>*)y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZsymm | FileCheck %s -check-prefix=cublasZsymm
// cublasZsymm: CUDA API:
// cublasZsymm-NEXT:   cublasZsymm(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasZsymm-NEXT:               upper_lower /*cublasFillMode_t*/, m /*int*/, n /*int*/,
// cublasZsymm-NEXT:               alpha /*const cuDoubleComplex **/, a /*const cuDoubleComplex **/,
// cublasZsymm-NEXT:               lda /*int*/, b /*const cuDoubleComplex **/, ldb /*int*/,
// cublasZsymm-NEXT:               beta /*const cuDoubleComplex **/, c /*cuDoubleComplex **/,
// cublasZsymm-NEXT:               ldc /*int*/);
// cublasZsymm-NEXT: Is migrated to:
// cublasZsymm-NEXT:   oneapi::mkl::blas::column_major::symm(*handle, left_right, upper_lower, m, n, dpct::get_value(alpha, *handle), (std::complex<double>*)a, lda, (std::complex<double>*)b, ldb, dpct::get_value(beta, *handle), (std::complex<double>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasGetStream | FileCheck %s -check-prefix=cublasGetStream
// cublasGetStream: CUDA API:
// cublasGetStream-NEXT:   cublasGetStream(handle /*cublasHandle_t*/, stream /*cudaStream_t **/);
// cublasGetStream-NEXT: Is migrated to:
// cublasGetStream-NEXT:   *stream = handle;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasScopy | FileCheck %s -check-prefix=cublasScopy
// cublasScopy: CUDA API:
// cublasScopy-NEXT:   cublasScopy(handle /*cublasHandle_t*/, n /*int*/, x /*const float **/,
// cublasScopy-NEXT:               incx /*int*/, y /*float **/, incy /*int*/);
// cublasScopy-NEXT: Is migrated to:
// cublasScopy-NEXT:   oneapi::mkl::blas::column_major::copy(*handle, n, x, incx, y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasGetMatrixAsync | FileCheck %s -check-prefix=cublasGetMatrixAsync
// cublasGetMatrixAsync: CUDA API:
// cublasGetMatrixAsync-NEXT:   cublasGetMatrixAsync(rows /*int*/, cols /*int*/, elementsize /*int*/,
// cublasGetMatrixAsync-NEXT:                        a /*const void **/, lda /*int*/, b /*void **/,
// cublasGetMatrixAsync-NEXT:                        ldb /*int*/, stream /*cudaStream_t*/);
// cublasGetMatrixAsync-NEXT: Is migrated to:
// cublasGetMatrixAsync-NEXT:   dpct::matrix_mem_copy((void*)b, (void*)a, ldb, lda, rows, cols, elementsize, dpct::automatic, *stream, true);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDtrmm | FileCheck %s -check-prefix=cublasDtrmm
// cublasDtrmm: CUDA API:
// cublasDtrmm-NEXT:   cublasDtrmm(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasDtrmm-NEXT:               upper_lower /*cublasFillMode_t*/, transa /*cublasOperation_t*/,
// cublasDtrmm-NEXT:               unit_diag /*cublasDiagType_t*/, m /*int*/, n /*int*/,
// cublasDtrmm-NEXT:               alpha /*const double **/, a /*const double **/, lda /*int*/,
// cublasDtrmm-NEXT:               b /*const double **/, ldb /*int*/, c /*double **/, ldc /*int*/);
// cublasDtrmm-NEXT: Is migrated to:
// cublasDtrmm-NEXT:   dpct::trmm(*handle, left_right, upper_lower, transa, unit_diag, m, n, alpha, a, lda, b, ldb, c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZher | FileCheck %s -check-prefix=cublasZher
// cublasZher: CUDA API:
// cublasZher-NEXT:   cublasZher(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZher-NEXT:              n /*int*/, alpha /*const double **/, x /*const cuDoubleComplex **/,
// cublasZher-NEXT:              incx /*int*/, a /*cuDoubleComplex **/, lda /*int*/);
// cublasZher-NEXT: Is migrated to:
// cublasZher-NEXT:   oneapi::mkl::blas::column_major::her(*handle, upper_lower, n, dpct::get_value(alpha, *handle), (std::complex<double>*)x, incx, (std::complex<double>*)a, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDgemmStridedBatched | FileCheck %s -check-prefix=cublasDgemmStridedBatched
// cublasDgemmStridedBatched: CUDA API:
// cublasDgemmStridedBatched-NEXT:   cublasDgemmStridedBatched(
// cublasDgemmStridedBatched-NEXT:       handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasDgemmStridedBatched-NEXT:       transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
// cublasDgemmStridedBatched-NEXT:       alpha /*const double **/, a /*const double **/, lda /*int*/,
// cublasDgemmStridedBatched-NEXT:       stridea /*long long int*/, b /*const double **/, ldb /*int*/,
// cublasDgemmStridedBatched-NEXT:       strideb /*long long int*/, beta /*const double **/, c /*double **/,
// cublasDgemmStridedBatched-NEXT:       ldc /*int*/, stridec /*long long int*/, group_count /*int*/);
// cublasDgemmStridedBatched-NEXT: Is migrated to:
// cublasDgemmStridedBatched-NEXT:   oneapi::mkl::blas::column_major::gemm_batch(*handle, transa, transb, m, n, k, dpct::get_value(alpha, *handle), a, lda, stridea, b, ldb, strideb, dpct::get_value(beta, *handle), c, ldc, stridec, group_count);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSdgmm | FileCheck %s -check-prefix=cublasSdgmm
// cublasSdgmm: CUDA API:
// cublasSdgmm-NEXT:   cublasSdgmm(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasSdgmm-NEXT:               m /*int*/, n /*int*/, a /*const float **/, lda /*int*/,
// cublasSdgmm-NEXT:               x /*const float **/, incx /*int*/, c /*float **/, ldc /*int*/);
// cublasSdgmm-NEXT: Is migrated to:
// cublasSdgmm-NEXT:   oneapi::mkl::blas::column_major::dgmm_batch(*handle, left_right, m, n, a, lda, 0, x, incx, 0, c, ldc, ldc * n, 1);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCsyrkx | FileCheck %s -check-prefix=cublasCsyrkx
// cublasCsyrkx: CUDA API:
// cublasCsyrkx-NEXT:   cublasCsyrkx(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCsyrkx-NEXT:                trans /*cublasOperation_t*/, n /*int*/, k /*int*/,
// cublasCsyrkx-NEXT:                alpha /*const cuComplex **/, a /*const cuComplex **/,
// cublasCsyrkx-NEXT:                lda /*int*/, b /*const cuComplex **/, ldb /*int*/,
// cublasCsyrkx-NEXT:                beta /*const cuComplex **/, c /*cuComplex **/, ldc /*int*/);
// cublasCsyrkx-NEXT: Is migrated to:
// cublasCsyrkx-NEXT:   dpct::syrk(*handle, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSetPointerMode | FileCheck %s -check-prefix=cublasSetPointerMode
// cublasSetPointerMode: CUDA API:
// cublasSetPointerMode-NEXT:   cublasSetPointerMode(handle /*cublasHandle_t*/,
// cublasSetPointerMode-NEXT:                        host_device /*cublasPointerMode_t*/);
// cublasSetPointerMode-NEXT: The API is Removed.

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCgerc | FileCheck %s -check-prefix=cublasCgerc
// cublasCgerc: CUDA API:
// cublasCgerc-NEXT:   cublasCgerc(handle /*cublasHandle_t*/, m /*int*/, n /*int*/,
// cublasCgerc-NEXT:               alpha /*const cuComplex **/, x /*const cuComplex **/,
// cublasCgerc-NEXT:               incx /*int*/, y /*const cuComplex **/, incy /*int*/,
// cublasCgerc-NEXT:               a /*cuComplex **/, lda /*int*/);
// cublasCgerc-NEXT: Is migrated to:
// cublasCgerc-NEXT:   oneapi::mkl::blas::column_major::gerc(*handle, m, n, dpct::get_value(alpha, *handle), (std::complex<float>*)x, incx, (std::complex<float>*)y, incy, (std::complex<float>*)a, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZgbmv | FileCheck %s -check-prefix=cublasZgbmv
// cublasZgbmv: CUDA API:
// cublasZgbmv-NEXT:   cublasZgbmv(handle /*cublasHandle_t*/, trans /*cublasOperation_t*/, m /*int*/,
// cublasZgbmv-NEXT:               n /*int*/, kl /*int*/, ku /*int*/,
// cublasZgbmv-NEXT:               alpha /*const cuDoubleComplex **/, a /*const cuDoubleComplex **/,
// cublasZgbmv-NEXT:               lda /*int*/, x /*const cuDoubleComplex **/, incx /*int*/,
// cublasZgbmv-NEXT:               beta /*const cuDoubleComplex **/, y /*cuDoubleComplex **/,
// cublasZgbmv-NEXT:               incy /*int*/);
// cublasZgbmv-NEXT: Is migrated to:
// cublasZgbmv-NEXT:   oneapi::mkl::blas::column_major::gbmv(*handle, trans, m, n, kl, ku, dpct::get_value(alpha, *handle), (std::complex<double>*)a, lda, (std::complex<double>*)x, incx, dpct::get_value(beta, *handle), (std::complex<double>*)y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSspr2 | FileCheck %s -check-prefix=cublasSspr2
// cublasSspr2: CUDA API:
// cublasSspr2-NEXT:   cublasSspr2(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasSspr2-NEXT:               n /*int*/, alpha /*const float **/, x /*const float **/,
// cublasSspr2-NEXT:               incx /*int*/, y /*const float **/, incy /*int*/, a /*float **/);
// cublasSspr2-NEXT: Is migrated to:
// cublasSspr2-NEXT:   oneapi::mkl::blas::column_major::spr2(*handle, upper_lower, n, dpct::get_value(alpha, *handle), x, incx, y, incy, a);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZhpmv | FileCheck %s -check-prefix=cublasZhpmv
// cublasZhpmv: CUDA API:
// cublasZhpmv-NEXT:   cublasZhpmv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZhpmv-NEXT:               n /*int*/, alpha /*const cuDoubleComplex **/,
// cublasZhpmv-NEXT:               a /*const cuDoubleComplex **/, x /*const cuDoubleComplex **/,
// cublasZhpmv-NEXT:               incx /*int*/, beta /*const cuDoubleComplex **/,
// cublasZhpmv-NEXT:               y /*cuDoubleComplex **/, incy /*int*/);
// cublasZhpmv-NEXT: Is migrated to:
// cublasZhpmv-NEXT:   oneapi::mkl::blas::column_major::hpmv(*handle, upper_lower, n, dpct::get_value(alpha, *handle), (std::complex<double>*)a, (std::complex<double>*)x, incx, dpct::get_value(beta, *handle), (std::complex<double>*)y, incy);

