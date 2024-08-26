// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDsyr | FileCheck %s -check-prefix=cublasDsyr
// cublasDsyr: CUDA API:
// cublasDsyr-NEXT:   cublasDsyr(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDsyr-NEXT:              n /*int*/, alpha /*const double **/, x /*const double **/,
// cublasDsyr-NEXT:              incx /*int*/, a /*double **/, lda /*int*/);
// cublasDsyr-NEXT: Is migrated to:
// cublasDsyr-NEXT:   oneapi::mkl::blas::column_major::syr(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), x, incx, a, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZhpr2 | FileCheck %s -check-prefix=cublasZhpr2
// cublasZhpr2: CUDA API:
// cublasZhpr2-NEXT:   cublasZhpr2(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZhpr2-NEXT:               n /*int*/, alpha /*const cuDoubleComplex **/,
// cublasZhpr2-NEXT:               x /*const cuDoubleComplex **/, incx /*int*/,
// cublasZhpr2-NEXT:               y /*const cuDoubleComplex **/, incy /*int*/,
// cublasZhpr2-NEXT:               a /*cuDoubleComplex **/);
// cublasZhpr2-NEXT: Is migrated to:
// cublasZhpr2-NEXT:   oneapi::mkl::blas::column_major::hpr2(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<double>*)x, incx, (std::complex<double>*)y, incy, (std::complex<double>*)a);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZtpmv | FileCheck %s -check-prefix=cublasZtpmv
// cublasZtpmv: CUDA API:
// cublasZtpmv-NEXT:   cublasZtpmv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZtpmv-NEXT:               trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasZtpmv-NEXT:               n /*int*/, a /*const cuDoubleComplex **/, x /*cuDoubleComplex **/,
// cublasZtpmv-NEXT:               incx /*int*/);
// cublasZtpmv-NEXT: Is migrated to:
// cublasZtpmv-NEXT:   oneapi::mkl::blas::column_major::tpmv(handle->get_queue(), upper_lower, trans, unit_nonunit, n, (std::complex<double>*)a, (std::complex<double>*)x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZherkx | FileCheck %s -check-prefix=cublasZherkx
// cublasZherkx: CUDA API:
// cublasZherkx-NEXT:   cublasZherkx(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZherkx-NEXT:                trans /*cublasOperation_t*/, n /*int*/, k /*int*/,
// cublasZherkx-NEXT:                alpha /*const cuDoubleComplex **/, a /*const cuDoubleComplex **/,
// cublasZherkx-NEXT:                lda /*int*/, b /*const cuDoubleComplex **/, ldb /*int*/,
// cublasZherkx-NEXT:                beta /*const double **/, c /*cuDoubleComplex **/, ldc /*int*/);
// cublasZherkx-NEXT: Is migrated to:
// cublasZherkx-NEXT:   dpct::blas::herk(handle, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasChbmv | FileCheck %s -check-prefix=cublasChbmv
// cublasChbmv: CUDA API:
// cublasChbmv-NEXT:   cublasChbmv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasChbmv-NEXT:               n /*int*/, k /*int*/, alpha /*const cuComplex **/,
// cublasChbmv-NEXT:               a /*const cuComplex **/, lda /*int*/, x /*const cuComplex **/,
// cublasChbmv-NEXT:               incx /*int*/, beta /*const cuComplex **/, y /*cuComplex **/,
// cublasChbmv-NEXT:               incy /*int*/);
// cublasChbmv-NEXT: Is migrated to:
// cublasChbmv-NEXT:   oneapi::mkl::blas::column_major::hbmv(handle->get_queue(), upper_lower, n, k, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)a, lda, (std::complex<float>*)x, incx, dpct::get_value(beta, handle->get_queue()), (std::complex<float>*)y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSrotm | FileCheck %s -check-prefix=cublasSrotm
// cublasSrotm: CUDA API:
// cublasSrotm-NEXT:   cublasSrotm(handle /*cublasHandle_t*/, n /*int*/, x /*float **/, incx /*int*/,
// cublasSrotm-NEXT:               y /*float **/, incy /*int*/, param /*const float **/);
// cublasSrotm-NEXT: Is migrated to:
// cublasSrotm-NEXT:   [&]() {
// cublasSrotm-NEXT:   dpct::blas::wrapper_float_in res_wrapper_ct6(handle->get_queue(), param, 5);
// cublasSrotm-NEXT:   oneapi::mkl::blas::column_major::rotm(handle->get_queue(), n, x, incx, y, incy, res_wrapper_ct6.get_ptr());
// cublasSrotm-NEXT:   return 0;
// cublasSrotm-NEXT:   }();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCher | FileCheck %s -check-prefix=cublasCher
// cublasCher: CUDA API:
// cublasCher-NEXT:   cublasCher(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCher-NEXT:              n /*int*/, alpha /*const float **/, x /*const cuComplex **/,
// cublasCher-NEXT:              incx /*int*/, a /*cuComplex **/, lda /*int*/);
// cublasCher-NEXT: Is migrated to:
// cublasCher-NEXT:   oneapi::mkl::blas::column_major::her(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)x, incx, (std::complex<float>*)a, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSgemm | FileCheck %s -check-prefix=cublasSgemm
// cublasSgemm: CUDA API:
// cublasSgemm-NEXT:   cublasSgemm(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasSgemm-NEXT:               transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
// cublasSgemm-NEXT:               alpha /*const float **/, a /*const float **/, lda /*int*/,
// cublasSgemm-NEXT:               b /*const float **/, ldb /*int*/, beta /*const float **/,
// cublasSgemm-NEXT:               c /*float **/, ldc /*int*/);
// cublasSgemm-NEXT: Is migrated to:
// cublasSgemm-NEXT:   oneapi::mkl::blas::column_major::gemm(handle->get_queue(), transa, transb, m, n, k, dpct::get_value(alpha, handle->get_queue()), a, lda, b, ldb, dpct::get_value(beta, handle->get_queue()), c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSsyrk | FileCheck %s -check-prefix=cublasSsyrk
// cublasSsyrk: CUDA API:
// cublasSsyrk-NEXT:   cublasSsyrk(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasSsyrk-NEXT:               trans /*cublasOperation_t*/, n /*int*/, k /*int*/,
// cublasSsyrk-NEXT:               alpha /*const float **/, a /*const float **/, lda /*int*/,
// cublasSsyrk-NEXT:               beta /*const float **/, c /*float **/, ldc /*int*/);
// cublasSsyrk-NEXT: Is migrated to:
// cublasSsyrk-NEXT:   oneapi::mkl::blas::column_major::syrk(handle->get_queue(), upper_lower, trans, n, k, dpct::get_value(alpha, handle->get_queue()), a, lda, dpct::get_value(beta, handle->get_queue()), c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZhemm | FileCheck %s -check-prefix=cublasZhemm
// cublasZhemm: CUDA API:
// cublasZhemm-NEXT:   cublasZhemm(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasZhemm-NEXT:               upper_lower /*cublasFillMode_t*/, m /*int*/, n /*int*/,
// cublasZhemm-NEXT:               alpha /*const cuDoubleComplex **/, a /*const cuDoubleComplex **/,
// cublasZhemm-NEXT:               lda /*int*/, b /*const cuDoubleComplex **/, ldb /*int*/,
// cublasZhemm-NEXT:               beta /*const cuDoubleComplex **/, c /*cuDoubleComplex **/,
// cublasZhemm-NEXT:               ldc /*int*/);
// cublasZhemm-NEXT: Is migrated to:
// cublasZhemm-NEXT:   oneapi::mkl::blas::column_major::hemm(handle->get_queue(), left_right, upper_lower, m, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<double>*)a, lda, (std::complex<double>*)b, ldb, dpct::get_value(beta, handle->get_queue()), (std::complex<double>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSetMatrixAsync | FileCheck %s -check-prefix=cublasSetMatrixAsync
// cublasSetMatrixAsync: CUDA API:
// cublasSetMatrixAsync-NEXT:   cublasSetMatrixAsync(rows /*int*/, cols /*int*/, elementsize /*int*/,
// cublasSetMatrixAsync-NEXT:                        a /*const void **/, lda /*int*/, b /*void **/,
// cublasSetMatrixAsync-NEXT:                        ldb /*int*/, stream /*cudaStream_t*/);
// cublasSetMatrixAsync-NEXT: Is migrated to:
// cublasSetMatrixAsync-NEXT:   dpct::blas::matrix_mem_copy(b, a, ldb, lda, rows, cols, elementsize, dpct::cs::memcpy_direction::automatic, *stream, true);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSaxpy | FileCheck %s -check-prefix=cublasSaxpy
// cublasSaxpy: CUDA API:
// cublasSaxpy-NEXT:   cublasSaxpy(handle /*cublasHandle_t*/, n /*int*/, alpha /*const float **/,
// cublasSaxpy-NEXT:               x /*const float **/, incx /*int*/, y /*float **/, incy /*int*/);
// cublasSaxpy-NEXT: Is migrated to:
// cublasSaxpy-NEXT:   oneapi::mkl::blas::column_major::axpy(handle->get_queue(), n, dpct::get_value(alpha, handle->get_queue()), x, incx, y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSetAtomicsMode | FileCheck %s -check-prefix=cublasSetAtomicsMode
// cublasSetAtomicsMode: CUDA API:
// cublasSetAtomicsMode-NEXT:   cublasSetAtomicsMode(handle /*cublasHandle_t*/,
// cublasSetAtomicsMode-NEXT:                        atomics /*cublasAtomicsMode_t*/);
// cublasSetAtomicsMode-NEXT: The API is Removed.

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDsymv | FileCheck %s -check-prefix=cublasDsymv
// cublasDsymv: CUDA API:
// cublasDsymv-NEXT:   cublasDsymv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDsymv-NEXT:               n /*int*/, alpha /*const double **/, a /*const double **/,
// cublasDsymv-NEXT:               lda /*int*/, x /*const double **/, incx /*int*/,
// cublasDsymv-NEXT:               beta /*const double **/, y /*double **/, incy /*int*/);
// cublasDsymv-NEXT: Is migrated to:
// cublasDsymv-NEXT:   oneapi::mkl::blas::column_major::symv(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), a, lda, x, incx, dpct::get_value(beta, handle->get_queue()), y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDsymm | FileCheck %s -check-prefix=cublasDsymm
// cublasDsymm: CUDA API:
// cublasDsymm-NEXT:   cublasDsymm(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasDsymm-NEXT:               upper_lower /*cublasFillMode_t*/, m /*int*/, n /*int*/,
// cublasDsymm-NEXT:               alpha /*const double **/, a /*const double **/, lda /*int*/,
// cublasDsymm-NEXT:               b /*const double **/, ldb /*int*/, beta /*const double **/,
// cublasDsymm-NEXT:               c /*double **/, ldc /*int*/);
// cublasDsymm-NEXT: Is migrated to:
// cublasDsymm-NEXT:   oneapi::mkl::blas::column_major::symm(handle->get_queue(), left_right, upper_lower, m, n, dpct::get_value(alpha, handle->get_queue()), a, lda, b, ldb, dpct::get_value(beta, handle->get_queue()), c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasChpr2 | FileCheck %s -check-prefix=cublasChpr2
// cublasChpr2: CUDA API:
// cublasChpr2-NEXT:   cublasChpr2(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasChpr2-NEXT:               n /*int*/, alpha /*const cuComplex **/, x /*const cuComplex **/,
// cublasChpr2-NEXT:               incx /*int*/, y /*const cuComplex **/, incy /*int*/,
// cublasChpr2-NEXT:               a /*cuComplex **/);
// cublasChpr2-NEXT: Is migrated to:
// cublasChpr2-NEXT:   oneapi::mkl::blas::column_major::hpr2(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)x, incx, (std::complex<float>*)y, incy, (std::complex<float>*)a);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCtbsv | FileCheck %s -check-prefix=cublasCtbsv
// cublasCtbsv: CUDA API:
// cublasCtbsv-NEXT:   cublasCtbsv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCtbsv-NEXT:               trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasCtbsv-NEXT:               n /*int*/, k /*int*/, a /*const cuComplex **/, lda /*int*/,
// cublasCtbsv-NEXT:               x /*cuComplex **/, incx /*int*/);
// cublasCtbsv-NEXT: Is migrated to:
// cublasCtbsv-NEXT:   oneapi::mkl::blas::column_major::tbsv(handle->get_queue(), upper_lower, trans, unit_nonunit, n, k, (std::complex<float>*)a, lda, (std::complex<float>*)x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDsbmv | FileCheck %s -check-prefix=cublasDsbmv
// cublasDsbmv: CUDA API:
// cublasDsbmv-NEXT:   cublasDsbmv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDsbmv-NEXT:               n /*int*/, k /*int*/, alpha /*const double **/,
// cublasDsbmv-NEXT:               a /*const double **/, lda /*int*/, x /*const double **/,
// cublasDsbmv-NEXT:               incx /*int*/, beta /*const double **/, y /*double **/,
// cublasDsbmv-NEXT:               incy /*int*/);
// cublasDsbmv-NEXT: Is migrated to:
// cublasDsbmv-NEXT:   oneapi::mkl::blas::column_major::sbmv(handle->get_queue(), upper_lower, n, k, dpct::get_value(alpha, handle->get_queue()), a, lda, x, incx, dpct::get_value(beta, handle->get_queue()), y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCgemmEx | FileCheck %s -check-prefix=cublasCgemmEx
// cublasCgemmEx: CUDA API:
// cublasCgemmEx-NEXT:   cublasCgemmEx(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasCgemmEx-NEXT:                 transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
// cublasCgemmEx-NEXT:                 alpha /*const cuComplex **/, a /*const void **/,
// cublasCgemmEx-NEXT:                 atype /*cudaDataType*/, lda /*int*/, b /*const void **/,
// cublasCgemmEx-NEXT:                 btype /*cudaDataType*/, ldb /*int*/, beta /*const cuComplex **/,
// cublasCgemmEx-NEXT:                 c /*void **/, ctype /*cudaDataType*/, ldc /*int*/);
// cublasCgemmEx-NEXT: Is migrated to:
// cublasCgemmEx-NEXT:   dpct::blas::gemm(handle, transa, transb, m, n, k, alpha, a, atype, lda, b, btype, ldb, beta, c, ctype, ldc, dpct::library_data_t::complex_float);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasIsamax | FileCheck %s -check-prefix=cublasIsamax
// cublasIsamax: CUDA API:
// cublasIsamax-NEXT:   cublasIsamax(handle /*cublasHandle_t*/, n /*int*/, x /*const float **/,
// cublasIsamax-NEXT:                incx /*int*/, res /*int **/);
// cublasIsamax-NEXT: Is migrated to (with the option --no-dry-pattern):
// cublasIsamax-NEXT:   [&]() {
// cublasIsamax-NEXT:   dpct::blas::wrapper_int_to_int64_out res_wrapper_ct4(handle->get_queue(), res);
// cublasIsamax-NEXT:   oneapi::mkl::blas::column_major::iamax(handle->get_queue(), n, x, incx, res_wrapper_ct4.get_ptr(), oneapi::mkl::index_base::one);
// cublasIsamax-NEXT:   return 0;
// cublasIsamax-NEXT:   }();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasGetMatrix | FileCheck %s -check-prefix=cublasGetMatrix
// cublasGetMatrix: CUDA API:
// cublasGetMatrix-NEXT:   cublasGetMatrix(rows /*int*/, cols /*int*/, elementsize /*int*/,
// cublasGetMatrix-NEXT:                   a /*const void **/, lda /*int*/, b /*void **/, ldb /*int*/);
// cublasGetMatrix-NEXT: Is migrated to:
// cublasGetMatrix-NEXT:   dpct::blas::matrix_mem_copy(b, a, ldb, lda, rows, cols, elementsize);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZgeam | FileCheck %s -check-prefix=cublasZgeam
// cublasZgeam: CUDA API:
// cublasZgeam-NEXT:   cublasZgeam(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasZgeam-NEXT:               transb /*cublasOperation_t*/, m /*int*/, n /*int*/,
// cublasZgeam-NEXT:               alpha /*const cuDoubleComplex **/, a /*const cuDoubleComplex **/,
// cublasZgeam-NEXT:               lda /*int*/, beta /*const cuDoubleComplex **/,
// cublasZgeam-NEXT:               b /*const cuDoubleComplex **/, ldb /*int*/,
// cublasZgeam-NEXT:               c /*cuDoubleComplex **/, ldc /*int*/);
// cublasZgeam-NEXT: Is migrated to:
// cublasZgeam-NEXT:   oneapi::mkl::blas::column_major::omatadd(handle->get_queue(), transa, transb, m, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<double>*)a, lda, dpct::get_value(beta, handle->get_queue()), (std::complex<double>*)b, ldb, (std::complex<double>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCsyr2 | FileCheck %s -check-prefix=cublasCsyr2
// cublasCsyr2: CUDA API:
// cublasCsyr2-NEXT:   cublasCsyr2(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCsyr2-NEXT:               n /*int*/, alpha /*const cuComplex **/, x /*const cuComplex **/,
// cublasCsyr2-NEXT:               incx /*int*/, y /*const cuComplex **/, incy /*int*/,
// cublasCsyr2-NEXT:               a /*cuComplex **/, lda /*int*/);
// cublasCsyr2-NEXT: Is migrated to:
// cublasCsyr2-NEXT:   oneapi::mkl::blas::column_major::syr2(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)x, incx, (std::complex<float>*)y, incy, (std::complex<float>*)a, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZsyrkx | FileCheck %s -check-prefix=cublasZsyrkx
// cublasZsyrkx: CUDA API:
// cublasZsyrkx-NEXT:   cublasZsyrkx(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZsyrkx-NEXT:                trans /*cublasOperation_t*/, n /*int*/, k /*int*/,
// cublasZsyrkx-NEXT:                alpha /*const cuDoubleComplex **/, a /*const cuDoubleComplex **/,
// cublasZsyrkx-NEXT:                lda /*int*/, b /*const cuDoubleComplex **/, ldb /*int*/,
// cublasZsyrkx-NEXT:                beta /*const cuDoubleComplex **/, c /*cuDoubleComplex **/,
// cublasZsyrkx-NEXT:                ldc /*int*/);
// cublasZsyrkx-NEXT: Is migrated to:
// cublasZsyrkx-NEXT:   dpct::blas::syrk(handle, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZsyr2 | FileCheck %s -check-prefix=cublasZsyr2
// cublasZsyr2: CUDA API:
// cublasZsyr2-NEXT:   cublasZsyr2(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZsyr2-NEXT:               n /*int*/, alpha /*const cuDoubleComplex **/,
// cublasZsyr2-NEXT:               x /*const cuDoubleComplex **/, incx /*int*/,
// cublasZsyr2-NEXT:               y /*const cuDoubleComplex **/, incy /*int*/,
// cublasZsyr2-NEXT:               a /*cuDoubleComplex **/, lda /*int*/);
// cublasZsyr2-NEXT: Is migrated to:
// cublasZsyr2-NEXT:   oneapi::mkl::blas::column_major::syr2(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<double>*)x, incx, (std::complex<double>*)y, incy, (std::complex<double>*)a, lda);
