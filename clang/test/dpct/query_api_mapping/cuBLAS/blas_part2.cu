// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDotEx | FileCheck %s -check-prefix=cublasDotEx
// cublasDotEx: CUDA API:
// cublasDotEx-NEXT:   cublasDotEx(handle /*cublasHandle_t*/, n /*int*/, x /*const void **/,
// cublasDotEx-NEXT:               xtype /*cudaDataType*/, incx /*int*/, y /*const void **/,
// cublasDotEx-NEXT:               ytype /*cudaDataType*/, incy /*int*/, res /*void **/,
// cublasDotEx-NEXT:               restype /*cudaDataType*/, computetype /*cudaDataType*/);
// cublasDotEx-NEXT: Is migrated to:
// cublasDotEx-NEXT:   dpct::dot(*handle, n, x, xtype, incx, y, ytype, incy, res, restype);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDtbmv | FileCheck %s -check-prefix=cublasDtbmv
// cublasDtbmv: CUDA API:
// cublasDtbmv-NEXT:   cublasDtbmv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDtbmv-NEXT:               trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasDtbmv-NEXT:               n /*int*/, k /*int*/, a /*const double **/, lda /*int*/,
// cublasDtbmv-NEXT:               x /*double **/, incx /*int*/);
// cublasDtbmv-NEXT: Is migrated to:
// cublasDtbmv-NEXT:   oneapi::mkl::blas::column_major::tbmv(*handle, upper_lower, trans, unit_nonunit, n, k, a, lda, x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCgemv | FileCheck %s -check-prefix=cublasCgemv
// cublasCgemv: CUDA API:
// cublasCgemv-NEXT:   cublasCgemv(handle /*cublasHandle_t*/, trans /*cublasOperation_t*/, m /*int*/,
// cublasCgemv-NEXT:               n /*int*/, alpha /*const cuComplex **/, a /*const cuComplex **/,
// cublasCgemv-NEXT:               lda /*int*/, x /*const cuComplex **/, incx /*int*/,
// cublasCgemv-NEXT:               beta /*const cuComplex **/, y /*cuComplex **/, incy /*int*/);
// cublasCgemv-NEXT: Is migrated to:
// cublasCgemv-NEXT:   oneapi::mkl::blas::column_major::gemv(*handle, trans, m, n, dpct::get_value(alpha, *handle), (std::complex<float>*)a, lda, (std::complex<float>*)x, incx, dpct::get_value(beta, *handle), (std::complex<float>*)y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCsyr | FileCheck %s -check-prefix=cublasCsyr
// cublasCsyr: CUDA API:
// cublasCsyr-NEXT:   cublasCsyr(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCsyr-NEXT:              n /*int*/, alpha /*const cuComplex **/, x /*const cuComplex **/,
// cublasCsyr-NEXT:              incx /*int*/, a /*cuComplex **/, lda /*int*/);
// cublasCsyr-NEXT: Is migrated to:
// cublasCsyr-NEXT:   oneapi::mkl::blas::column_major::syr(*handle, upper_lower, n, dpct::get_value(alpha, *handle), (std::complex<float>*)x, incx, (std::complex<float>*)a, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZcopy | FileCheck %s -check-prefix=cublasZcopy
// cublasZcopy: CUDA API:
// cublasZcopy-NEXT:   cublasZcopy(handle /*cublasHandle_t*/, n /*int*/,
// cublasZcopy-NEXT:               x /*const cuDoubleComplex **/, incx /*int*/,
// cublasZcopy-NEXT:               y /*cuDoubleComplex **/, incy /*int*/);
// cublasZcopy-NEXT: Is migrated to:
// cublasZcopy-NEXT:   oneapi::mkl::blas::column_major::copy(*handle, n, (std::complex<double>*)x, incx, (std::complex<double>*)y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDspmv | FileCheck %s -check-prefix=cublasDspmv
// cublasDspmv: CUDA API:
// cublasDspmv-NEXT:   cublasDspmv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDspmv-NEXT:               n /*int*/, alpha /*const double **/, a /*const double **/,
// cublasDspmv-NEXT:               x /*const double **/, incx /*int*/, beta /*const double **/,
// cublasDspmv-NEXT:               y /*double **/, incy /*int*/);
// cublasDspmv-NEXT: Is migrated to:
// cublasDspmv-NEXT:   oneapi::mkl::blas::column_major::spmv(*handle, upper_lower, n, dpct::get_value(alpha, *handle), a, x, incx, dpct::get_value(beta, *handle), y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZtrsm | FileCheck %s -check-prefix=cublasZtrsm
// cublasZtrsm: CUDA API:
// cublasZtrsm-NEXT:   cublasZtrsm(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasZtrsm-NEXT:               upper_lower /*cublasFillMode_t*/, transa /*cublasOperation_t*/,
// cublasZtrsm-NEXT:               unit_diag /*cublasDiagType_t*/, m /*int*/, n /*int*/,
// cublasZtrsm-NEXT:               alpha /*const cuDoubleComplex **/, a /*const cuDoubleComplex **/,
// cublasZtrsm-NEXT:               lda /*int*/, b /*cuDoubleComplex **/, ldb /*int*/);
// cublasZtrsm-NEXT: Is migrated to:
// cublasZtrsm-NEXT:   oneapi::mkl::blas::column_major::trsm(*handle, left_right, upper_lower, transa, unit_diag, m, n, dpct::get_value(alpha, *handle), (std::complex<double>*)a, lda, (std::complex<double>*)b, ldb);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSasum | FileCheck %s -check-prefix=cublasSasum
// cublasSasum: CUDA API:
// cublasSasum-NEXT:   cublasSasum(handle /*cublasHandle_t*/, n /*int*/, x /*const float **/,
// cublasSasum-NEXT:               incx /*int*/, res /*float **/);
// cublasSasum-NEXT: Is migrated to:
// cublasSasum-NEXT:   float* res_temp_ptr_ct{{[0-9]+}} = res;
// cublasSasum-NEXT:   if(sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasSasum-NEXT:     res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<float>(1, dpct::get_in_order_queue());
// cublasSasum-NEXT:   }
// cublasSasum-NEXT:   oneapi::mkl::blas::column_major::asum(*handle, n, x, incx, res_temp_ptr_ct{{[0-9]+}});
// cublasSasum-NEXT:   if(sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasSasum-NEXT:     handle->wait();
// cublasSasum-NEXT:     *res = *res_temp_ptr_ct{{[0-9]+}};
// cublasSasum-NEXT:     sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_in_order_queue());
// cublasSasum-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCsyr2k | FileCheck %s -check-prefix=cublasCsyr2k
// cublasCsyr2k: CUDA API:
// cublasCsyr2k-NEXT:   cublasCsyr2k(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCsyr2k-NEXT:                trans /*cublasOperation_t*/, n /*int*/, k /*int*/,
// cublasCsyr2k-NEXT:                alpha /*const cuComplex **/, a /*const cuComplex **/,
// cublasCsyr2k-NEXT:                lda /*int*/, b /*const cuComplex **/, ldb /*int*/,
// cublasCsyr2k-NEXT:                beta /*const cuComplex **/, c /*cuComplex **/, ldc /*int*/);
// cublasCsyr2k-NEXT: Is migrated to:
// cublasCsyr2k-NEXT:   oneapi::mkl::blas::column_major::syr2k(*handle, upper_lower, trans, n, k, dpct::get_value(alpha, *handle), (std::complex<float>*)a, lda, (std::complex<float>*)b, ldb, dpct::get_value(beta, *handle), (std::complex<float>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZtrsv | FileCheck %s -check-prefix=cublasZtrsv
// cublasZtrsv: CUDA API:
// cublasZtrsv-NEXT:   cublasZtrsv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZtrsv-NEXT:               trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasZtrsv-NEXT:               n /*int*/, a /*const cuDoubleComplex **/, lda /*int*/,
// cublasZtrsv-NEXT:               x /*cuDoubleComplex **/, incx /*int*/);
// cublasZtrsv-NEXT: Is migrated to:
// cublasZtrsv-NEXT:   oneapi::mkl::blas::column_major::trsv(*handle, upper_lower, trans, unit_nonunit, n, (std::complex<double>*)a, lda, (std::complex<double>*)x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCrotg | FileCheck %s -check-prefix=cublasCrotg
// cublasCrotg: CUDA API:
// cublasCrotg-NEXT:   cublasCrotg(handle /*cublasHandle_t*/, a /*cuComplex **/, b /*cuComplex **/,
// cublasCrotg-NEXT:               c /*float **/, s /*cuComplex **/);
// cublasCrotg-NEXT: Is migrated to:
// cublasCrotg-NEXT:   sycl::float2* a_ct{{[0-9]+}} = a;
// cublasCrotg-NEXT:   sycl::float2* b_ct{{[0-9]+}} = b;
// cublasCrotg-NEXT:   float* c_ct{{[0-9]+}} = c;
// cublasCrotg-NEXT:   sycl::float2* s_ct{{[0-9]+}} = s;
// cublasCrotg-NEXT:   if(sycl::get_pointer_type(a, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(a, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasCrotg-NEXT:     a_ct{{[0-9]+}} = sycl::malloc_shared<sycl::float2>(3, dpct::get_in_order_queue());
// cublasCrotg-NEXT:     c_ct{{[0-9]+}} = sycl::malloc_shared<float>(1, dpct::get_in_order_queue());
// cublasCrotg-NEXT:     b_ct{{[0-9]+}} = a_ct{{[0-9]+}} + 1;
// cublasCrotg-NEXT:     s_ct{{[0-9]+}} = a_ct{{[0-9]+}} + 2;
// cublasCrotg-NEXT:     *a_ct{{[0-9]+}} = *a;
// cublasCrotg-NEXT:     *b_ct{{[0-9]+}} = *b;
// cublasCrotg-NEXT:     *c_ct{{[0-9]+}} = *c;
// cublasCrotg-NEXT:     *s_ct{{[0-9]+}} = *s;
// cublasCrotg-NEXT:   }
// cublasCrotg-NEXT:   oneapi::mkl::blas::column_major::rotg(*handle, (std::complex<float>*)a_ct{{[0-9]+}}, (std::complex<float>*)b_ct{{[0-9]+}}, c_ct{{[0-9]+}}, (std::complex<float>*)s_ct{{[0-9]+}});
// cublasCrotg-NEXT:   if(sycl::get_pointer_type(a, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(a, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasCrotg-NEXT:     handle->wait();
// cublasCrotg-NEXT:     *a = *a_ct{{[0-9]+}};
// cublasCrotg-NEXT:     *b = *b_ct{{[0-9]+}};
// cublasCrotg-NEXT:     *c = *c_ct{{[0-9]+}};
// cublasCrotg-NEXT:     *s = *s_ct{{[0-9]+}};
// cublasCrotg-NEXT:     sycl::free(a_ct{{[0-9]+}}, dpct::get_in_order_queue());
// cublasCrotg-NEXT:     sycl::free(c_ct{{[0-9]+}}, dpct::get_in_order_queue());
// cublasCrotg-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZtbsv | FileCheck %s -check-prefix=cublasZtbsv
// cublasZtbsv: CUDA API:
// cublasZtbsv-NEXT:   cublasZtbsv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZtbsv-NEXT:               trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasZtbsv-NEXT:               n /*int*/, k /*int*/, a /*const cuDoubleComplex **/, lda /*int*/,
// cublasZtbsv-NEXT:               x /*cuDoubleComplex **/, incx /*int*/);
// cublasZtbsv-NEXT: Is migrated to:
// cublasZtbsv-NEXT:   oneapi::mkl::blas::column_major::tbsv(*handle, upper_lower, trans, unit_nonunit, n, k, (std::complex<double>*)a, lda, (std::complex<double>*)x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCscal | FileCheck %s -check-prefix=cublasCscal
// cublasCscal: CUDA API:
// cublasCscal-NEXT:   cublasCscal(handle /*cublasHandle_t*/, n /*int*/, alpha /*const cuComplex **/,
// cublasCscal-NEXT:               x /*cuComplex **/, incx /*int*/);
// cublasCscal-NEXT: Is migrated to:
// cublasCscal-NEXT:   oneapi::mkl::blas::column_major::scal(*handle, n, dpct::get_value(alpha, *handle), (std::complex<float>*)x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDsyrk | FileCheck %s -check-prefix=cublasDsyrk
// cublasDsyrk: CUDA API:
// cublasDsyrk-NEXT:   cublasDsyrk(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDsyrk-NEXT:               trans /*cublasOperation_t*/, n /*int*/, k /*int*/,
// cublasDsyrk-NEXT:               alpha /*const double **/, a /*const double **/, lda /*int*/,
// cublasDsyrk-NEXT:               beta /*const double **/, c /*double **/, ldc /*int*/);
// cublasDsyrk-NEXT: Is migrated to:
// cublasDsyrk-NEXT:   oneapi::mkl::blas::column_major::syrk(*handle, upper_lower, trans, n, k, dpct::get_value(alpha, *handle), a, lda, dpct::get_value(beta, *handle), c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDswap | FileCheck %s -check-prefix=cublasDswap
// cublasDswap: CUDA API:
// cublasDswap-NEXT:   cublasDswap(handle /*cublasHandle_t*/, n /*int*/, x /*double **/,
// cublasDswap-NEXT:               incx /*int*/, y /*double **/, incy /*int*/);
// cublasDswap-NEXT: Is migrated to:
// cublasDswap-NEXT:   oneapi::mkl::blas::column_major::swap(*handle, n, x, incx, y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZher2 | FileCheck %s -check-prefix=cublasZher2
// cublasZher2: CUDA API:
// cublasZher2-NEXT:   cublasZher2(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZher2-NEXT:               n /*int*/, alpha /*const cuDoubleComplex **/,
// cublasZher2-NEXT:               x /*const cuDoubleComplex **/, incx /*int*/,
// cublasZher2-NEXT:               y /*const cuDoubleComplex **/, incy /*int*/,
// cublasZher2-NEXT:               a /*cuDoubleComplex **/, lda /*int*/);
// cublasZher2-NEXT: Is migrated to:
// cublasZher2-NEXT:   oneapi::mkl::blas::column_major::her2(*handle, upper_lower, n, dpct::get_value(alpha, *handle), (std::complex<double>*)x, incx, (std::complex<double>*)y, incy, (std::complex<double>*)a, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCtbmv | FileCheck %s -check-prefix=cublasCtbmv
// cublasCtbmv: CUDA API:
// cublasCtbmv-NEXT:   cublasCtbmv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCtbmv-NEXT:               trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasCtbmv-NEXT:               n /*int*/, k /*int*/, a /*const cuComplex **/, lda /*int*/,
// cublasCtbmv-NEXT:               x /*cuComplex **/, incx /*int*/);
// cublasCtbmv-NEXT: Is migrated to:
// cublasCtbmv-NEXT:   oneapi::mkl::blas::column_major::tbmv(*handle, upper_lower, trans, unit_nonunit, n, k, (std::complex<float>*)a, lda, (std::complex<float>*)x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZgemm | FileCheck %s -check-prefix=cublasZgemm
// cublasZgemm: CUDA API:
// cublasZgemm-NEXT:   cublasZgemm(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasZgemm-NEXT:               transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
// cublasZgemm-NEXT:               alpha /*const cuDoubleComplex **/, a /*const cuDoubleComplex **/,
// cublasZgemm-NEXT:               lda /*int*/, b /*const cuDoubleComplex **/, ldb /*int*/,
// cublasZgemm-NEXT:               beta /*const cuDoubleComplex **/, c /*cuDoubleComplex **/,
// cublasZgemm-NEXT:               ldc /*int*/);
// cublasZgemm-NEXT: Is migrated to:
// cublasZgemm-NEXT:   oneapi::mkl::blas::column_major::gemm(*handle, transa, transb, m, n, k, dpct::get_value(alpha, *handle), (std::complex<double>*)a, lda, (std::complex<double>*)b, ldb, dpct::get_value(beta, *handle), (std::complex<double>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCher2k | FileCheck %s -check-prefix=cublasCher2k
// cublasCher2k: CUDA API:
// cublasCher2k-NEXT:   cublasCher2k(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCher2k-NEXT:                trans /*cublasOperation_t*/, n /*int*/, k /*int*/,
// cublasCher2k-NEXT:                alpha /*const cuComplex **/, a /*const cuComplex **/,
// cublasCher2k-NEXT:                lda /*int*/, b /*const cuComplex **/, ldb /*int*/,
// cublasCher2k-NEXT:                beta /*const float **/, c /*cuComplex **/, ldc /*int*/);
// cublasCher2k-NEXT: Is migrated to:
// cublasCher2k-NEXT:   oneapi::mkl::blas::column_major::her2k(*handle, upper_lower, trans, n, k, dpct::get_value(alpha, *handle), (std::complex<float>*)a, lda, (std::complex<float>*)b, ldb, dpct::get_value(beta, *handle), (std::complex<float>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCcopy | FileCheck %s -check-prefix=cublasCcopy
// cublasCcopy: CUDA API:
// cublasCcopy-NEXT:   cublasCcopy(handle /*cublasHandle_t*/, n /*int*/, x /*const cuComplex **/,
// cublasCcopy-NEXT:               incx /*int*/, y /*cuComplex **/, incy /*int*/);
// cublasCcopy-NEXT: Is migrated to:
// cublasCcopy-NEXT:   oneapi::mkl::blas::column_major::copy(*handle, n, (std::complex<float>*)x, incx, (std::complex<float>*)y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSscal | FileCheck %s -check-prefix=cublasSscal
// cublasSscal: CUDA API:
// cublasSscal-NEXT:   cublasSscal(handle /*cublasHandle_t*/, n /*int*/, alpha /*const float **/,
// cublasSscal-NEXT:               x /*float **/, incx /*int*/);
// cublasSscal-NEXT: Is migrated to:
// cublasSscal-NEXT:   oneapi::mkl::blas::column_major::scal(*handle, n, dpct::get_value(alpha, *handle), x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasIzamax | FileCheck %s -check-prefix=cublasIzamax
// cublasIzamax: CUDA API:
// cublasIzamax-NEXT:   cublasIzamax(handle /*cublasHandle_t*/, n /*int*/,
// cublasIzamax-NEXT:                x /*const cuDoubleComplex **/, incx /*int*/, res /*int **/);
// cublasIzamax-NEXT: Is migrated to (with the option --no-dry-pattern):
// cublasIzamax-NEXT:   [&]() {
// cublasIzamax-NEXT:   dpct::blas::result_memory_t<std::int64_t, int> res(res);
// cublasIzamax-NEXT:   oneapi::mkl::blas::column_major::iamax(*handle, n, (std::complex<double>*)x, incx, res.get_memory(), oneapi::mkl::index_base::one);
// cublasIzamax-NEXT:   return 0;
// cublasIzamax-NEXT:   }();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasScnrm2 | FileCheck %s -check-prefix=cublasScnrm2
// cublasScnrm2: CUDA API:
// cublasScnrm2-NEXT:   cublasScnrm2(handle /*cublasHandle_t*/, n /*int*/, x /*const cuComplex **/,
// cublasScnrm2-NEXT:                incx /*int*/, res /*float **/);
// cublasScnrm2-NEXT: Is migrated to:
// cublasScnrm2-NEXT:   float* res_temp_ptr_ct{{[0-9]+}} = res;
// cublasScnrm2-NEXT:   if(sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasScnrm2-NEXT:     res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<float>(1, dpct::get_in_order_queue());
// cublasScnrm2-NEXT:   }
// cublasScnrm2-NEXT:   oneapi::mkl::blas::column_major::nrm2(*handle, n, (std::complex<float>*)x, incx, res_temp_ptr_ct{{[0-9]+}});
// cublasScnrm2-NEXT:   if(sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasScnrm2-NEXT:     handle->wait();
// cublasScnrm2-NEXT:     *res = *res_temp_ptr_ct{{[0-9]+}};
// cublasScnrm2-NEXT:     sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_in_order_queue());
// cublasScnrm2-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSrot | FileCheck %s -check-prefix=cublasSrot
// cublasSrot: CUDA API:
// cublasSrot-NEXT:   cublasSrot(handle /*cublasHandle_t*/, n /*int*/, x /*float **/, incx /*int*/,
// cublasSrot-NEXT:              y /*float **/, incy /*int*/, c /*const float **/,
// cublasSrot-NEXT:              s /*const float **/);
// cublasSrot-NEXT: Is migrated to:
// cublasSrot-NEXT:   oneapi::mkl::blas::column_major::rot(*handle, n, x, incx, y, incy, dpct::get_value(c, *handle), dpct::get_value(s, *handle));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZtpsv | FileCheck %s -check-prefix=cublasZtpsv
// cublasZtpsv: CUDA API:
// cublasZtpsv-NEXT:   cublasZtpsv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZtpsv-NEXT:               trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasZtpsv-NEXT:               n /*int*/, a /*const cuDoubleComplex **/, x /*cuDoubleComplex **/,
// cublasZtpsv-NEXT:               incx /*int*/);
// cublasZtpsv-NEXT: Is migrated to:
// cublasZtpsv-NEXT:   oneapi::mkl::blas::column_major::tpsv(*handle, upper_lower, trans, unit_nonunit, n, (std::complex<double>*)a, (std::complex<double>*)x, incx);
