// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDgeam | FileCheck %s -check-prefix=cublasDgeam
// cublasDgeam: CUDA API:
// cublasDgeam-NEXT:   cublasDgeam(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasDgeam-NEXT:               transb /*cublasOperation_t*/, m /*int*/, n /*int*/,
// cublasDgeam-NEXT:               alpha /*const double **/, a /*const double **/, lda /*int*/,
// cublasDgeam-NEXT:               beta /*const double **/, b /*const double **/, ldb /*int*/,
// cublasDgeam-NEXT:               c /*double **/, ldc /*int*/);
// cublasDgeam-NEXT: Is migrated to:
// cublasDgeam-NEXT:   oneapi::mkl::blas::column_major::omatadd(handle->get_queue(), transa, transb, m, n, dpct::get_value(alpha, handle->get_queue()), a, lda, dpct::get_value(beta, handle->get_queue()), b, ldb, c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDotcEx | FileCheck %s -check-prefix=cublasDotcEx
// cublasDotcEx: CUDA API:
// cublasDotcEx-NEXT:   cublasDotcEx(handle /*cublasHandle_t*/, n /*int*/, x /*const void **/,
// cublasDotcEx-NEXT:                xtype /*cudaDataType*/, incx /*int*/, y /*const void **/,
// cublasDotcEx-NEXT:                ytype /*cudaDataType*/, incy /*int*/, res /*void **/,
// cublasDotcEx-NEXT:                restype /*cudaDataType*/, computetype /*cudaDataType*/);
// cublasDotcEx-NEXT: Is migrated to:
// cublasDotcEx-NEXT:   dpct::blas::dotc(handle, n, x, xtype, incx, y, ytype, incy, res, restype);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSsyr2 | FileCheck %s -check-prefix=cublasSsyr2
// cublasSsyr2: CUDA API:
// cublasSsyr2-NEXT:   cublasSsyr2(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasSsyr2-NEXT:               n /*int*/, alpha /*const float **/, x /*const float **/,
// cublasSsyr2-NEXT:               incx /*int*/, y /*const float **/, incy /*int*/, a /*float **/,
// cublasSsyr2-NEXT:               lda /*int*/);
// cublasSsyr2-NEXT: Is migrated to:
// cublasSsyr2-NEXT:   oneapi::mkl::blas::column_major::syr2(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), x, incx, y, incy, a, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasChemv | FileCheck %s -check-prefix=cublasChemv
// cublasChemv: CUDA API:
// cublasChemv-NEXT:   cublasChemv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasChemv-NEXT:               n /*int*/, alpha /*const cuComplex **/, a /*const cuComplex **/,
// cublasChemv-NEXT:               lda /*int*/, x /*const cuComplex **/, incx /*int*/,
// cublasChemv-NEXT:               beta /*const cuComplex **/, y /*cuComplex **/, incy /*int*/);
// cublasChemv-NEXT: Is migrated to:
// cublasChemv-NEXT:   oneapi::mkl::blas::column_major::hemv(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)a, lda, (std::complex<float>*)x, incx, dpct::get_value(beta, handle->get_queue()), (std::complex<float>*)y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDsyrkx | FileCheck %s -check-prefix=cublasDsyrkx
// cublasDsyrkx: CUDA API:
// cublasDsyrkx-NEXT:   cublasDsyrkx(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDsyrkx-NEXT:                trans /*cublasOperation_t*/, n /*int*/, k /*int*/,
// cublasDsyrkx-NEXT:                alpha /*const double **/, a /*const double **/, lda /*int*/,
// cublasDsyrkx-NEXT:                b /*const double **/, ldb /*int*/, beta /*const double **/,
// cublasDsyrkx-NEXT:                c /*double **/, ldc /*int*/);
// cublasDsyrkx-NEXT: Is migrated to:
// cublasDsyrkx-NEXT:   dpct::blas::syrk(handle, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZtbmv | FileCheck %s -check-prefix=cublasZtbmv
// cublasZtbmv: CUDA API:
// cublasZtbmv-NEXT:   cublasZtbmv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZtbmv-NEXT:               trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasZtbmv-NEXT:               n /*int*/, k /*int*/, a /*const cuDoubleComplex **/, lda /*int*/,
// cublasZtbmv-NEXT:               x /*cuDoubleComplex **/, incx /*int*/);
// cublasZtbmv-NEXT: Is migrated to:
// cublasZtbmv-NEXT:   oneapi::mkl::blas::column_major::tbmv(handle->get_queue(), upper_lower, trans, unit_nonunit, n, k, (std::complex<double>*)a, lda, (std::complex<double>*)x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasHgemmStridedBatched | FileCheck %s -check-prefix=cublasHgemmStridedBatched
// cublasHgemmStridedBatched: CUDA API:
// cublasHgemmStridedBatched-NEXT:   cublasHgemmStridedBatched(
// cublasHgemmStridedBatched-NEXT:       handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasHgemmStridedBatched-NEXT:       transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
// cublasHgemmStridedBatched-NEXT:       alpha /*const __half **/, a /*const __half **/, lda /*int*/,
// cublasHgemmStridedBatched-NEXT:       stridea /*long long int*/, b /*const __half **/, ldb /*int*/,
// cublasHgemmStridedBatched-NEXT:       strideb /*long long int*/, beta /*const __half **/, c /*__half **/,
// cublasHgemmStridedBatched-NEXT:       ldc /*int*/, stridec /*long long int*/, group_count /*int*/);
// cublasHgemmStridedBatched-NEXT: Is migrated to:
// cublasHgemmStridedBatched-NEXT:   oneapi::mkl::blas::column_major::gemm_batch(handle->get_queue(), transa, transb, m, n, k, dpct::get_value(alpha, handle->get_queue()), a, lda, stridea, b, ldb, strideb, dpct::get_value(beta, handle->get_queue()), c, ldc, stridec, group_count);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDzasum | FileCheck %s -check-prefix=cublasDzasum
// cublasDzasum: CUDA API:
// cublasDzasum-NEXT:   cublasDzasum(handle /*cublasHandle_t*/, n /*int*/,
// cublasDzasum-NEXT:                x /*const cuDoubleComplex **/, incx /*int*/, res /*double **/);
// cublasDzasum-NEXT: Is migrated to:
// cublasDzasum-NEXT:   [&]() {
// cublasDzasum-NEXT:   dpct::blas::wrapper_double_out res_wrapper_ct4(handle->get_queue(), res);
// cublasDzasum-NEXT:   oneapi::mkl::blas::column_major::asum(handle->get_queue(), n, (std::complex<double>*)x, incx, res_wrapper_ct4.get_ptr());
// cublasDzasum-NEXT:   return 0;
// cublasDzasum-NEXT:   }();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZgemm3m | FileCheck %s -check-prefix=cublasZgemm3m
// cublasZgemm3m: CUDA API:
// cublasZgemm3m-NEXT:   cublasZgemm3m(
// cublasZgemm3m-NEXT:       handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasZgemm3m-NEXT:       transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
// cublasZgemm3m-NEXT:       alpha /*const cuDoubleComplex **/, a /*const cuDoubleComplex **/,
// cublasZgemm3m-NEXT:       lda /*int*/, b /*const cuDoubleComplex **/, ldb /*int*/,
// cublasZgemm3m-NEXT:       beta /*const cuDoubleComplex **/, c /*cuDoubleComplex **/, ldc /*int*/);
// cublasZgemm3m-NEXT: Is migrated to:
// cublasZgemm3m-NEXT:   oneapi::mkl::blas::column_major::gemm(handle->get_queue(), transa, transb, m, n, k, dpct::get_value(alpha, handle->get_queue()), (std::complex<double>*)a, lda, (std::complex<double>*)b, ldb, dpct::get_value(beta, handle->get_queue()), (std::complex<double>*)c, ldc, oneapi::mkl::blas::compute_mode::complex_3m);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSspmv | FileCheck %s -check-prefix=cublasSspmv
// cublasSspmv: CUDA API:
// cublasSspmv-NEXT:   cublasSspmv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasSspmv-NEXT:               n /*int*/, alpha /*const float **/, a /*const float **/,
// cublasSspmv-NEXT:               x /*const float **/, incx /*int*/, beta /*const float **/,
// cublasSspmv-NEXT:               y /*float **/, incy /*int*/);
// cublasSspmv-NEXT: Is migrated to:
// cublasSspmv-NEXT:   oneapi::mkl::blas::column_major::spmv(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), a, x, incx, dpct::get_value(beta, handle->get_queue()), y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCsrot | FileCheck %s -check-prefix=cublasCsrot
// cublasCsrot: CUDA API:
// cublasCsrot-NEXT:   cublasCsrot(handle /*cublasHandle_t*/, n /*int*/, x /*cuComplex **/,
// cublasCsrot-NEXT:               incx /*int*/, y /*cuComplex **/, incy /*int*/,
// cublasCsrot-NEXT:               c /*const float **/, s /*const float **/);
// cublasCsrot-NEXT: Is migrated to:
// cublasCsrot-NEXT:   oneapi::mkl::blas::column_major::rot(handle->get_queue(), n, (std::complex<float>*)x, incx, (std::complex<float>*)y, incy, dpct::get_value(c, handle->get_queue()), dpct::get_value(s, handle->get_queue()));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasGetVersion | FileCheck %s -check-prefix=cublasGetVersion
// cublasGetVersion: CUDA API:
// cublasGetVersion-NEXT:   cublasGetVersion(handle /*cublasHandle_t*/, ver /*int **/);
// cublasGetVersion-NEXT: Is migrated to:
// cublasGetVersion-NEXT:   dpct::mkl_get_version(dpct::version_field::major, ver);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasIcamin | FileCheck %s -check-prefix=cublasIcamin
// cublasIcamin: CUDA API:
// cublasIcamin-NEXT:   cublasIcamin(handle /*cublasHandle_t*/, n /*int*/, x /*const cuComplex **/,
// cublasIcamin-NEXT:                incx /*int*/, res /*int **/);
// cublasIcamin-NEXT: Is migrated to (with the option --no-dry-pattern):
// cublasIcamin-NEXT:   [&]() {
// cublasIcamin-NEXT:   dpct::blas::wrapper_int_to_int64_out res_wrapper_ct4(handle->get_queue(), res);
// cublasIcamin-NEXT:   oneapi::mkl::blas::column_major::iamin(handle->get_queue(), n, (std::complex<float>*)x, incx, res_wrapper_ct4.get_ptr(), oneapi::mkl::index_base::one);
// cublasIcamin-NEXT:   return 0;
// cublasIcamin-NEXT:   }();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasIcamax | FileCheck %s -check-prefix=cublasIcamax
// cublasIcamax: CUDA API:
// cublasIcamax-NEXT:   cublasIcamax(handle /*cublasHandle_t*/, n /*int*/, x /*const cuComplex **/,
// cublasIcamax-NEXT:                incx /*int*/, res /*int **/);
// cublasIcamax-NEXT: Is migrated to (with the option --no-dry-pattern):
// cublasIcamax-NEXT:   [&]() {
// cublasIcamax-NEXT:   dpct::blas::wrapper_int_to_int64_out res_wrapper_ct4(handle->get_queue(), res);
// cublasIcamax-NEXT:   oneapi::mkl::blas::column_major::iamax(handle->get_queue(), n, (std::complex<float>*)x, incx, res_wrapper_ct4.get_ptr(), oneapi::mkl::index_base::one);
// cublasIcamax-NEXT:   return 0;
// cublasIcamax-NEXT:   }();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDdgmm | FileCheck %s -check-prefix=cublasDdgmm
// cublasDdgmm: CUDA API:
// cublasDdgmm-NEXT:   cublasDdgmm(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasDdgmm-NEXT:               m /*int*/, n /*int*/, a /*const double **/, lda /*int*/,
// cublasDdgmm-NEXT:               x /*const double **/, incx /*int*/, c /*double **/, ldc /*int*/);
// cublasDdgmm-NEXT: Is migrated to:
// cublasDdgmm-NEXT:   oneapi::mkl::blas::column_major::dgmm(handle->get_queue(), left_right, m, n, a, lda, x, incx, c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasStrmv | FileCheck %s -check-prefix=cublasStrmv
// cublasStrmv: CUDA API:
// cublasStrmv-NEXT:   cublasStrmv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasStrmv-NEXT:               trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasStrmv-NEXT:               n /*int*/, a /*const float **/, lda /*int*/, x /*float **/,
// cublasStrmv-NEXT:               incx /*int*/);
// cublasStrmv-NEXT: Is migrated to:
// cublasStrmv-NEXT:   oneapi::mkl::blas::column_major::trmv(handle->get_queue(), upper_lower, trans, unit_nonunit, n, a, lda, x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCrot | FileCheck %s -check-prefix=cublasCrot
// cublasCrot: CUDA API:
// cublasCrot-NEXT:   cublasCrot(handle /*cublasHandle_t*/, n /*int*/, x /*cuComplex **/,
// cublasCrot-NEXT:              incx /*int*/, y /*cuComplex **/, incy /*int*/, c /*const float **/,
// cublasCrot-NEXT:              s /*const cuComplex **/);
// cublasCrot-NEXT: Is migrated to:
// cublasCrot-NEXT:   oneapi::mkl::blas::column_major::rot(handle->get_queue(), n, (std::complex<float>*)x, incx, (std::complex<float>*)y, incy, dpct::get_value(c, handle->get_queue()), dpct::get_value(s, handle->get_queue()));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCreate | FileCheck %s -check-prefix=cublasCreate
// cublasCreate: CUDA API:
// cublasCreate-NEXT:   cublasCreate(handle /*cublasHandle_t **/);
// cublasCreate-NEXT: Is migrated to:
// cublasCreate-NEXT:   *handle = new dpct::blas::descriptor();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasChpmv | FileCheck %s -check-prefix=cublasChpmv
// cublasChpmv: CUDA API:
// cublasChpmv-NEXT:   cublasChpmv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasChpmv-NEXT:               n /*int*/, alpha /*const cuComplex **/, a /*const cuComplex **/,
// cublasChpmv-NEXT:               x /*const cuComplex **/, incx /*int*/, beta /*const cuComplex **/,
// cublasChpmv-NEXT:               y /*cuComplex **/, incy /*int*/);
// cublasChpmv-NEXT: Is migrated to:
// cublasChpmv-NEXT:   oneapi::mkl::blas::column_major::hpmv(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)a, (std::complex<float>*)x, incx, dpct::get_value(beta, handle->get_queue()), (std::complex<float>*)y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSnrm2 | FileCheck %s -check-prefix=cublasSnrm2
// cublasSnrm2: CUDA API:
// cublasSnrm2-NEXT:   cublasSnrm2(handle /*cublasHandle_t*/, n /*int*/, x /*const float **/,
// cublasSnrm2-NEXT:               incx /*int*/, res /*float **/);
// cublasSnrm2-NEXT: Is migrated to:
// cublasSnrm2-NEXT:   [&]() {
// cublasSnrm2-NEXT:   dpct::blas::wrapper_float_out res_wrapper_ct4(handle->get_queue(), res);
// cublasSnrm2-NEXT:   oneapi::mkl::blas::column_major::nrm2(handle->get_queue(), n, x, incx, res_wrapper_ct4.get_ptr());
// cublasSnrm2-NEXT:   return 0;
// cublasSnrm2-NEXT:   }();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZgemmStridedBatched | FileCheck %s -check-prefix=cublasZgemmStridedBatched
// cublasZgemmStridedBatched: CUDA API:
// cublasZgemmStridedBatched-NEXT:   cublasZgemmStridedBatched(
// cublasZgemmStridedBatched-NEXT:       handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasZgemmStridedBatched-NEXT:       transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
// cublasZgemmStridedBatched-NEXT:       alpha /*const cuDoubleComplex **/, a /*const cuDoubleComplex **/,
// cublasZgemmStridedBatched-NEXT:       lda /*int*/, stridea /*long long int*/, b /*const cuDoubleComplex **/,
// cublasZgemmStridedBatched-NEXT:       ldb /*int*/, strideb /*long long int*/, beta /*const cuDoubleComplex **/,
// cublasZgemmStridedBatched-NEXT:       c /*cuDoubleComplex **/, ldc /*int*/, stridec /*long long int*/,
// cublasZgemmStridedBatched-NEXT:       group_count /*int*/);
// cublasZgemmStridedBatched-NEXT: Is migrated to:
// cublasZgemmStridedBatched-NEXT:   oneapi::mkl::blas::column_major::gemm_batch(handle->get_queue(), transa, transb, m, n, k, dpct::get_value(alpha, handle->get_queue()), (std::complex<double>*)a, lda, stridea, (std::complex<double>*)b, ldb, strideb, dpct::get_value(beta, handle->get_queue()), (std::complex<double>*)c, ldc, stridec, group_count);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZsyr | FileCheck %s -check-prefix=cublasZsyr
// cublasZsyr: CUDA API:
// cublasZsyr-NEXT:   cublasZsyr(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZsyr-NEXT:              n /*int*/, alpha /*const cuDoubleComplex **/,
// cublasZsyr-NEXT:              x /*const cuDoubleComplex **/, incx /*int*/,
// cublasZsyr-NEXT:              a /*cuDoubleComplex **/, lda /*int*/);
// cublasZsyr-NEXT: Is migrated to:
// cublasZsyr-NEXT:   oneapi::mkl::blas::column_major::syr(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<double>*)x, incx, (std::complex<double>*)a, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDtpsv | FileCheck %s -check-prefix=cublasDtpsv
// cublasDtpsv: CUDA API:
// cublasDtpsv-NEXT:   cublasDtpsv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDtpsv-NEXT:               trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasDtpsv-NEXT:               n /*int*/, a /*const double **/, x /*double **/, incx /*int*/);
// cublasDtpsv-NEXT: Is migrated to:
// cublasDtpsv-NEXT:   oneapi::mkl::blas::column_major::tpsv(handle->get_queue(), upper_lower, trans, unit_nonunit, n, a, x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDtrsv | FileCheck %s -check-prefix=cublasDtrsv
// cublasDtrsv: CUDA API:
// cublasDtrsv-NEXT:   cublasDtrsv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDtrsv-NEXT:               trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasDtrsv-NEXT:               n /*int*/, a /*const double **/, lda /*int*/, x /*double **/,
// cublasDtrsv-NEXT:               incx /*int*/);
// cublasDtrsv-NEXT: Is migrated to:
// cublasDtrsv-NEXT:   oneapi::mkl::blas::column_major::trsv(handle->get_queue(), upper_lower, trans, unit_nonunit, n, a, lda, x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDgbmv | FileCheck %s -check-prefix=cublasDgbmv
// cublasDgbmv: CUDA API:
// cublasDgbmv-NEXT:   cublasDgbmv(handle /*cublasHandle_t*/, trans /*cublasOperation_t*/, m /*int*/,
// cublasDgbmv-NEXT:               n /*int*/, kl /*int*/, ku /*int*/, alpha /*const double **/,
// cublasDgbmv-NEXT:               a /*const double **/, lda /*int*/, x /*const double **/,
// cublasDgbmv-NEXT:               incx /*int*/, beta /*const double **/, y /*double **/,
// cublasDgbmv-NEXT:               incy /*int*/);
// cublasDgbmv-NEXT: Is migrated to:
// cublasDgbmv-NEXT:   oneapi::mkl::blas::column_major::gbmv(handle->get_queue(), trans, m, n, kl, ku, dpct::get_value(alpha, handle->get_queue()), a, lda, x, incx, dpct::get_value(beta, handle->get_queue()), y, incy);
