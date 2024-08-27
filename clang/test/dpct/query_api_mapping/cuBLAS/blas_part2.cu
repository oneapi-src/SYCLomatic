// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDotEx | FileCheck %s -check-prefix=cublasDotEx
// cublasDotEx: CUDA API:
// cublasDotEx-NEXT:   cublasDotEx(handle /*cublasHandle_t*/, n /*int*/, x /*const void **/,
// cublasDotEx-NEXT:               xtype /*cudaDataType*/, incx /*int*/, y /*const void **/,
// cublasDotEx-NEXT:               ytype /*cudaDataType*/, incy /*int*/, res /*void **/,
// cublasDotEx-NEXT:               restype /*cudaDataType*/, computetype /*cudaDataType*/);
// cublasDotEx-NEXT: Is migrated to:
// cublasDotEx-NEXT:   dpct::blas::dot(handle, n, x, xtype, incx, y, ytype, incy, res, restype);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDtbmv | FileCheck %s -check-prefix=cublasDtbmv
// cublasDtbmv: CUDA API:
// cublasDtbmv-NEXT:   cublasDtbmv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDtbmv-NEXT:               trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasDtbmv-NEXT:               n /*int*/, k /*int*/, a /*const double **/, lda /*int*/,
// cublasDtbmv-NEXT:               x /*double **/, incx /*int*/);
// cublasDtbmv-NEXT: Is migrated to:
// cublasDtbmv-NEXT:   oneapi::mkl::blas::column_major::tbmv(handle->get_queue(), upper_lower, trans, unit_nonunit, n, k, a, lda, x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCgemv | FileCheck %s -check-prefix=cublasCgemv
// cublasCgemv: CUDA API:
// cublasCgemv-NEXT:   cublasCgemv(handle /*cublasHandle_t*/, trans /*cublasOperation_t*/, m /*int*/,
// cublasCgemv-NEXT:               n /*int*/, alpha /*const cuComplex **/, a /*const cuComplex **/,
// cublasCgemv-NEXT:               lda /*int*/, x /*const cuComplex **/, incx /*int*/,
// cublasCgemv-NEXT:               beta /*const cuComplex **/, y /*cuComplex **/, incy /*int*/);
// cublasCgemv-NEXT: Is migrated to:
// cublasCgemv-NEXT:   oneapi::mkl::blas::column_major::gemv(handle->get_queue(), trans, m, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)a, lda, (std::complex<float>*)x, incx, dpct::get_value(beta, handle->get_queue()), (std::complex<float>*)y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCsyr | FileCheck %s -check-prefix=cublasCsyr
// cublasCsyr: CUDA API:
// cublasCsyr-NEXT:   cublasCsyr(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCsyr-NEXT:              n /*int*/, alpha /*const cuComplex **/, x /*const cuComplex **/,
// cublasCsyr-NEXT:              incx /*int*/, a /*cuComplex **/, lda /*int*/);
// cublasCsyr-NEXT: Is migrated to:
// cublasCsyr-NEXT:   oneapi::mkl::blas::column_major::syr(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)x, incx, (std::complex<float>*)a, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZcopy | FileCheck %s -check-prefix=cublasZcopy
// cublasZcopy: CUDA API:
// cublasZcopy-NEXT:   cublasZcopy(handle /*cublasHandle_t*/, n /*int*/,
// cublasZcopy-NEXT:               x /*const cuDoubleComplex **/, incx /*int*/,
// cublasZcopy-NEXT:               y /*cuDoubleComplex **/, incy /*int*/);
// cublasZcopy-NEXT: Is migrated to:
// cublasZcopy-NEXT:   oneapi::mkl::blas::column_major::copy(handle->get_queue(), n, (std::complex<double>*)x, incx, (std::complex<double>*)y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDspmv | FileCheck %s -check-prefix=cublasDspmv
// cublasDspmv: CUDA API:
// cublasDspmv-NEXT:   cublasDspmv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDspmv-NEXT:               n /*int*/, alpha /*const double **/, a /*const double **/,
// cublasDspmv-NEXT:               x /*const double **/, incx /*int*/, beta /*const double **/,
// cublasDspmv-NEXT:               y /*double **/, incy /*int*/);
// cublasDspmv-NEXT: Is migrated to:
// cublasDspmv-NEXT:   oneapi::mkl::blas::column_major::spmv(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), a, x, incx, dpct::get_value(beta, handle->get_queue()), y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZtrsm | FileCheck %s -check-prefix=cublasZtrsm
// cublasZtrsm: CUDA API:
// cublasZtrsm-NEXT:   cublasZtrsm(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasZtrsm-NEXT:               upper_lower /*cublasFillMode_t*/, transa /*cublasOperation_t*/,
// cublasZtrsm-NEXT:               unit_diag /*cublasDiagType_t*/, m /*int*/, n /*int*/,
// cublasZtrsm-NEXT:               alpha /*const cuDoubleComplex **/, a /*const cuDoubleComplex **/,
// cublasZtrsm-NEXT:               lda /*int*/, b /*cuDoubleComplex **/, ldb /*int*/);
// cublasZtrsm-NEXT: Is migrated to:
// cublasZtrsm-NEXT:   oneapi::mkl::blas::column_major::trsm(handle->get_queue(), left_right, upper_lower, transa, unit_diag, m, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<double>*)a, lda, (std::complex<double>*)b, ldb);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSasum | FileCheck %s -check-prefix=cublasSasum
// cublasSasum: CUDA API:
// cublasSasum-NEXT:   cublasSasum(handle /*cublasHandle_t*/, n /*int*/, x /*const float **/,
// cublasSasum-NEXT:               incx /*int*/, res /*float **/);
// cublasSasum-NEXT: Is migrated to:
// cublasSasum-NEXT:   [&]() {
// cublasSasum-NEXT:   dpct::blas::wrapper_float_out res_wrapper_ct4(handle->get_queue(), res);
// cublasSasum-NEXT:   oneapi::mkl::blas::column_major::asum(handle->get_queue(), n, x, incx, res_wrapper_ct4.get_ptr());
// cublasSasum-NEXT:   return 0;
// cublasSasum-NEXT:   }();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCsyr2k | FileCheck %s -check-prefix=cublasCsyr2k
// cublasCsyr2k: CUDA API:
// cublasCsyr2k-NEXT:   cublasCsyr2k(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCsyr2k-NEXT:                trans /*cublasOperation_t*/, n /*int*/, k /*int*/,
// cublasCsyr2k-NEXT:                alpha /*const cuComplex **/, a /*const cuComplex **/,
// cublasCsyr2k-NEXT:                lda /*int*/, b /*const cuComplex **/, ldb /*int*/,
// cublasCsyr2k-NEXT:                beta /*const cuComplex **/, c /*cuComplex **/, ldc /*int*/);
// cublasCsyr2k-NEXT: Is migrated to:
// cublasCsyr2k-NEXT:   oneapi::mkl::blas::column_major::syr2k(handle->get_queue(), upper_lower, trans, n, k, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)a, lda, (std::complex<float>*)b, ldb, dpct::get_value(beta, handle->get_queue()), (std::complex<float>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZtrsv | FileCheck %s -check-prefix=cublasZtrsv
// cublasZtrsv: CUDA API:
// cublasZtrsv-NEXT:   cublasZtrsv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZtrsv-NEXT:               trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasZtrsv-NEXT:               n /*int*/, a /*const cuDoubleComplex **/, lda /*int*/,
// cublasZtrsv-NEXT:               x /*cuDoubleComplex **/, incx /*int*/);
// cublasZtrsv-NEXT: Is migrated to:
// cublasZtrsv-NEXT:   oneapi::mkl::blas::column_major::trsv(handle->get_queue(), upper_lower, trans, unit_nonunit, n, (std::complex<double>*)a, lda, (std::complex<double>*)x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCrotg | FileCheck %s -check-prefix=cublasCrotg
// cublasCrotg: CUDA API:
// cublasCrotg-NEXT:   cublasCrotg(handle /*cublasHandle_t*/, a /*cuComplex **/, b /*cuComplex **/,
// cublasCrotg-NEXT:               c /*float **/, s /*cuComplex **/);
// cublasCrotg-NEXT: Is migrated to:
// cublasCrotg-NEXT:   [&]() {
// cublasCrotg-NEXT:   dpct::blas::wrapper_float2_inout res_wrapper_ct1(handle->get_queue(), a);
// cublasCrotg-NEXT:   dpct::blas::wrapper_float2_inout res_wrapper_ct2(handle->get_queue(), b);
// cublasCrotg-NEXT:   dpct::blas::wrapper_float_out res_wrapper_ct3(handle->get_queue(), c);
// cublasCrotg-NEXT:   dpct::blas::wrapper_float2_out res_wrapper_ct4(handle->get_queue(), s);
// cublasCrotg-NEXT:   oneapi::mkl::blas::column_major::rotg(handle->get_queue(), (std::complex<float>*)res_wrapper_ct1.get_ptr(), (std::complex<float>*)res_wrapper_ct2.get_ptr(), res_wrapper_ct3.get_ptr(), (std::complex<float>*)res_wrapper_ct4.get_ptr());
// cublasCrotg-NEXT:   return 0;
// cublasCrotg-NEXT:   }();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZtbsv | FileCheck %s -check-prefix=cublasZtbsv
// cublasZtbsv: CUDA API:
// cublasZtbsv-NEXT:   cublasZtbsv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZtbsv-NEXT:               trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasZtbsv-NEXT:               n /*int*/, k /*int*/, a /*const cuDoubleComplex **/, lda /*int*/,
// cublasZtbsv-NEXT:               x /*cuDoubleComplex **/, incx /*int*/);
// cublasZtbsv-NEXT: Is migrated to:
// cublasZtbsv-NEXT:   oneapi::mkl::blas::column_major::tbsv(handle->get_queue(), upper_lower, trans, unit_nonunit, n, k, (std::complex<double>*)a, lda, (std::complex<double>*)x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCscal | FileCheck %s -check-prefix=cublasCscal
// cublasCscal: CUDA API:
// cublasCscal-NEXT:   cublasCscal(handle /*cublasHandle_t*/, n /*int*/, alpha /*const cuComplex **/,
// cublasCscal-NEXT:               x /*cuComplex **/, incx /*int*/);
// cublasCscal-NEXT: Is migrated to:
// cublasCscal-NEXT:   oneapi::mkl::blas::column_major::scal(handle->get_queue(), n, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDsyrk | FileCheck %s -check-prefix=cublasDsyrk
// cublasDsyrk: CUDA API:
// cublasDsyrk-NEXT:   cublasDsyrk(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDsyrk-NEXT:               trans /*cublasOperation_t*/, n /*int*/, k /*int*/,
// cublasDsyrk-NEXT:               alpha /*const double **/, a /*const double **/, lda /*int*/,
// cublasDsyrk-NEXT:               beta /*const double **/, c /*double **/, ldc /*int*/);
// cublasDsyrk-NEXT: Is migrated to:
// cublasDsyrk-NEXT:   oneapi::mkl::blas::column_major::syrk(handle->get_queue(), upper_lower, trans, n, k, dpct::get_value(alpha, handle->get_queue()), a, lda, dpct::get_value(beta, handle->get_queue()), c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDswap | FileCheck %s -check-prefix=cublasDswap
// cublasDswap: CUDA API:
// cublasDswap-NEXT:   cublasDswap(handle /*cublasHandle_t*/, n /*int*/, x /*double **/,
// cublasDswap-NEXT:               incx /*int*/, y /*double **/, incy /*int*/);
// cublasDswap-NEXT: Is migrated to:
// cublasDswap-NEXT:   oneapi::mkl::blas::column_major::swap(handle->get_queue(), n, x, incx, y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZher2 | FileCheck %s -check-prefix=cublasZher2
// cublasZher2: CUDA API:
// cublasZher2-NEXT:   cublasZher2(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZher2-NEXT:               n /*int*/, alpha /*const cuDoubleComplex **/,
// cublasZher2-NEXT:               x /*const cuDoubleComplex **/, incx /*int*/,
// cublasZher2-NEXT:               y /*const cuDoubleComplex **/, incy /*int*/,
// cublasZher2-NEXT:               a /*cuDoubleComplex **/, lda /*int*/);
// cublasZher2-NEXT: Is migrated to:
// cublasZher2-NEXT:   oneapi::mkl::blas::column_major::her2(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<double>*)x, incx, (std::complex<double>*)y, incy, (std::complex<double>*)a, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCtbmv | FileCheck %s -check-prefix=cublasCtbmv
// cublasCtbmv: CUDA API:
// cublasCtbmv-NEXT:   cublasCtbmv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCtbmv-NEXT:               trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasCtbmv-NEXT:               n /*int*/, k /*int*/, a /*const cuComplex **/, lda /*int*/,
// cublasCtbmv-NEXT:               x /*cuComplex **/, incx /*int*/);
// cublasCtbmv-NEXT: Is migrated to:
// cublasCtbmv-NEXT:   oneapi::mkl::blas::column_major::tbmv(handle->get_queue(), upper_lower, trans, unit_nonunit, n, k, (std::complex<float>*)a, lda, (std::complex<float>*)x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZgemm | FileCheck %s -check-prefix=cublasZgemm
// cublasZgemm: CUDA API:
// cublasZgemm-NEXT:   cublasZgemm(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasZgemm-NEXT:               transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
// cublasZgemm-NEXT:               alpha /*const cuDoubleComplex **/, a /*const cuDoubleComplex **/,
// cublasZgemm-NEXT:               lda /*int*/, b /*const cuDoubleComplex **/, ldb /*int*/,
// cublasZgemm-NEXT:               beta /*const cuDoubleComplex **/, c /*cuDoubleComplex **/,
// cublasZgemm-NEXT:               ldc /*int*/);
// cublasZgemm-NEXT: Is migrated to:
// cublasZgemm-NEXT:   oneapi::mkl::blas::column_major::gemm(handle->get_queue(), transa, transb, m, n, k, dpct::get_value(alpha, handle->get_queue()), (std::complex<double>*)a, lda, (std::complex<double>*)b, ldb, dpct::get_value(beta, handle->get_queue()), (std::complex<double>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCher2k | FileCheck %s -check-prefix=cublasCher2k
// cublasCher2k: CUDA API:
// cublasCher2k-NEXT:   cublasCher2k(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCher2k-NEXT:                trans /*cublasOperation_t*/, n /*int*/, k /*int*/,
// cublasCher2k-NEXT:                alpha /*const cuComplex **/, a /*const cuComplex **/,
// cublasCher2k-NEXT:                lda /*int*/, b /*const cuComplex **/, ldb /*int*/,
// cublasCher2k-NEXT:                beta /*const float **/, c /*cuComplex **/, ldc /*int*/);
// cublasCher2k-NEXT: Is migrated to:
// cublasCher2k-NEXT:   oneapi::mkl::blas::column_major::her2k(handle->get_queue(), upper_lower, trans, n, k, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)a, lda, (std::complex<float>*)b, ldb, dpct::get_value(beta, handle->get_queue()), (std::complex<float>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCcopy | FileCheck %s -check-prefix=cublasCcopy
// cublasCcopy: CUDA API:
// cublasCcopy-NEXT:   cublasCcopy(handle /*cublasHandle_t*/, n /*int*/, x /*const cuComplex **/,
// cublasCcopy-NEXT:               incx /*int*/, y /*cuComplex **/, incy /*int*/);
// cublasCcopy-NEXT: Is migrated to:
// cublasCcopy-NEXT:   oneapi::mkl::blas::column_major::copy(handle->get_queue(), n, (std::complex<float>*)x, incx, (std::complex<float>*)y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSscal | FileCheck %s -check-prefix=cublasSscal
// cublasSscal: CUDA API:
// cublasSscal-NEXT:   cublasSscal(handle /*cublasHandle_t*/, n /*int*/, alpha /*const float **/,
// cublasSscal-NEXT:               x /*float **/, incx /*int*/);
// cublasSscal-NEXT: Is migrated to:
// cublasSscal-NEXT:   oneapi::mkl::blas::column_major::scal(handle->get_queue(), n, dpct::get_value(alpha, handle->get_queue()), x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasIzamax | FileCheck %s -check-prefix=cublasIzamax
// cublasIzamax: CUDA API:
// cublasIzamax-NEXT:   cublasIzamax(handle /*cublasHandle_t*/, n /*int*/,
// cublasIzamax-NEXT:                x /*const cuDoubleComplex **/, incx /*int*/, res /*int **/);
// cublasIzamax-NEXT: Is migrated to (with the option --no-dry-pattern):
// cublasIzamax-NEXT:   [&]() {
// cublasIzamax-NEXT:   dpct::blas::wrapper_int_to_int64_out res_wrapper_ct4(handle->get_queue(), res);
// cublasIzamax-NEXT:   oneapi::mkl::blas::column_major::iamax(handle->get_queue(), n, (std::complex<double>*)x, incx, res_wrapper_ct4.get_ptr(), oneapi::mkl::index_base::one);
// cublasIzamax-NEXT:   return 0;
// cublasIzamax-NEXT:   }();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasScnrm2 | FileCheck %s -check-prefix=cublasScnrm2
// cublasScnrm2: CUDA API:
// cublasScnrm2-NEXT:   cublasScnrm2(handle /*cublasHandle_t*/, n /*int*/, x /*const cuComplex **/,
// cublasScnrm2-NEXT:                incx /*int*/, res /*float **/);
// cublasScnrm2-NEXT: Is migrated to:
// cublasScnrm2-NEXT:   [&]() {
// cublasScnrm2-NEXT:   dpct::blas::wrapper_float_out res_wrapper_ct4(handle->get_queue(), res);
// cublasScnrm2-NEXT:   oneapi::mkl::blas::column_major::nrm2(handle->get_queue(), n, (std::complex<float>*)x, incx, res_wrapper_ct4.get_ptr());
// cublasScnrm2-NEXT:   return 0;
// cublasScnrm2-NEXT:   }();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSrot | FileCheck %s -check-prefix=cublasSrot
// cublasSrot: CUDA API:
// cublasSrot-NEXT:   cublasSrot(handle /*cublasHandle_t*/, n /*int*/, x /*float **/, incx /*int*/,
// cublasSrot-NEXT:              y /*float **/, incy /*int*/, c /*const float **/,
// cublasSrot-NEXT:              s /*const float **/);
// cublasSrot-NEXT: Is migrated to:
// cublasSrot-NEXT:   oneapi::mkl::blas::column_major::rot(handle->get_queue(), n, x, incx, y, incy, dpct::get_value(c, handle->get_queue()), dpct::get_value(s, handle->get_queue()));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZtpsv | FileCheck %s -check-prefix=cublasZtpsv
// cublasZtpsv: CUDA API:
// cublasZtpsv-NEXT:   cublasZtpsv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZtpsv-NEXT:               trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasZtpsv-NEXT:               n /*int*/, a /*const cuDoubleComplex **/, x /*cuDoubleComplex **/,
// cublasZtpsv-NEXT:               incx /*int*/);
// cublasZtpsv-NEXT: Is migrated to:
// cublasZtpsv-NEXT:   oneapi::mkl::blas::column_major::tpsv(handle->get_queue(), upper_lower, trans, unit_nonunit, n, (std::complex<double>*)a, (std::complex<double>*)x, incx);
