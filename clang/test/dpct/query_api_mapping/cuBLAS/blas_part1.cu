// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZdscal | FileCheck %s -check-prefix=cublasZdscal
// cublasZdscal: CUDA API:
// cublasZdscal-NEXT:   cublasZdscal(handle /*cublasHandle_t*/, n /*int*/, alpha /*const double **/,
// cublasZdscal-NEXT:                x /*cuDoubleComplex **/, incx /*int*/);
// cublasZdscal-NEXT: Is migrated to:
// cublasZdscal-NEXT:   oneapi::mkl::blas::column_major::scal(handle->get_queue(), n, dpct::get_value(alpha, handle->get_queue()), (std::complex<double>*)x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasGetVectorAsync | FileCheck %s -check-prefix=cublasGetVectorAsync
// cublasGetVectorAsync: CUDA API:
// cublasGetVectorAsync-NEXT:   cublasGetVectorAsync(n /*int*/, elementsize /*int*/, from /*const void **/,
// cublasGetVectorAsync-NEXT:                        incx /*int*/, to /*void **/, incy /*int*/,
// cublasGetVectorAsync-NEXT:                        stream /*cudaStream_t*/);
// cublasGetVectorAsync-NEXT: Is migrated to:
// cublasGetVectorAsync-NEXT:   dpct::blas::matrix_mem_copy(to, from, incy, incx, 1, n, elementsize, dpct::cs::memcpy_direction::automatic, *stream, true);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZsyrk | FileCheck %s -check-prefix=cublasZsyrk
// cublasZsyrk: CUDA API:
// cublasZsyrk-NEXT:   cublasZsyrk(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZsyrk-NEXT:               trans /*cublasOperation_t*/, n /*int*/, k /*int*/,
// cublasZsyrk-NEXT:               alpha /*const cuDoubleComplex **/, a /*const cuDoubleComplex **/,
// cublasZsyrk-NEXT:               lda /*int*/, beta /*const cuDoubleComplex **/,
// cublasZsyrk-NEXT:               c /*cuDoubleComplex **/, ldc /*int*/);
// cublasZsyrk-NEXT: Is migrated to:
// cublasZsyrk-NEXT:   oneapi::mkl::blas::column_major::syrk(handle->get_queue(), upper_lower, trans, n, k, dpct::get_value(alpha, handle->get_queue()), (std::complex<double>*)a, lda, dpct::get_value(beta, handle->get_queue()), (std::complex<double>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDtrsm | FileCheck %s -check-prefix=cublasDtrsm
// cublasDtrsm: CUDA API:
// cublasDtrsm-NEXT:   cublasDtrsm(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasDtrsm-NEXT:               upper_lower /*cublasFillMode_t*/, transa /*cublasOperation_t*/,
// cublasDtrsm-NEXT:               unit_diag /*cublasDiagType_t*/, m /*int*/, n /*int*/,
// cublasDtrsm-NEXT:               alpha /*const double **/, a /*const double **/, lda /*int*/,
// cublasDtrsm-NEXT:               b /*double **/, ldb /*int*/);
// cublasDtrsm-NEXT: Is migrated to:
// cublasDtrsm-NEXT:   oneapi::mkl::blas::column_major::trsm(handle->get_queue(), left_right, upper_lower, transa, unit_diag, m, n, dpct::get_value(alpha, handle->get_queue()), a, lda, b, ldb);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCgemmStridedBatched | FileCheck %s -check-prefix=cublasCgemmStridedBatched
// cublasCgemmStridedBatched: CUDA API:
// cublasCgemmStridedBatched-NEXT:   cublasCgemmStridedBatched(
// cublasCgemmStridedBatched-NEXT:       handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasCgemmStridedBatched-NEXT:       transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
// cublasCgemmStridedBatched-NEXT:       alpha /*const cuComplex **/, a /*const cuComplex **/, lda /*int*/,
// cublasCgemmStridedBatched-NEXT:       stridea /*long long int*/, b /*const cuComplex **/, ldb /*int*/,
// cublasCgemmStridedBatched-NEXT:       strideb /*long long int*/, beta /*const cuComplex **/, c /*cuComplex **/,
// cublasCgemmStridedBatched-NEXT:       ldc /*int*/, stridec /*long long int*/, group_count /*int*/);
// cublasCgemmStridedBatched-NEXT: Is migrated to:
// cublasCgemmStridedBatched-NEXT:   oneapi::mkl::blas::column_major::gemm_batch(handle->get_queue(), transa, transb, m, n, k, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)a, lda, stridea, (std::complex<float>*)b, ldb, strideb, dpct::get_value(beta, handle->get_queue()), (std::complex<float>*)c, ldc, stridec, group_count);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasChemm | FileCheck %s -check-prefix=cublasChemm
// cublasChemm: CUDA API:
// cublasChemm-NEXT:   cublasChemm(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasChemm-NEXT:               upper_lower /*cublasFillMode_t*/, m /*int*/, n /*int*/,
// cublasChemm-NEXT:               alpha /*const cuComplex **/, a /*const cuComplex **/, lda /*int*/,
// cublasChemm-NEXT:               b /*const cuComplex **/, ldb /*int*/, beta /*const cuComplex **/,
// cublasChemm-NEXT:               c /*cuComplex **/, ldc /*int*/);
// cublasChemm-NEXT: Is migrated to:
// cublasChemm-NEXT:   oneapi::mkl::blas::column_major::hemm(handle->get_queue(), left_right, upper_lower, m, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)a, lda, (std::complex<float>*)b, ldb, dpct::get_value(beta, handle->get_queue()), (std::complex<float>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDrotg | FileCheck %s -check-prefix=cublasDrotg
// cublasDrotg: CUDA API:
// cublasDrotg-NEXT:   cublasDrotg(handle /*cublasHandle_t*/, a /*double **/, b /*double **/,
// cublasDrotg-NEXT:               c /*double **/, s /*double **/);
// cublasDrotg-NEXT: Is migrated to:
// cublasDrotg-NEXT:   [&]() {
// cublasDrotg-NEXT:   dpct::blas::wrapper_double_inout res_wrapper_ct1(handle->get_queue(), a);
// cublasDrotg-NEXT:   dpct::blas::wrapper_double_inout res_wrapper_ct2(handle->get_queue(), b);
// cublasDrotg-NEXT:   dpct::blas::wrapper_double_out res_wrapper_ct3(handle->get_queue(), c);
// cublasDrotg-NEXT:   dpct::blas::wrapper_double_out res_wrapper_ct4(handle->get_queue(), s);
// cublasDrotg-NEXT:   oneapi::mkl::blas::column_major::rotg(handle->get_queue(), res_wrapper_ct1.get_ptr(), res_wrapper_ct2.get_ptr(), res_wrapper_ct3.get_ptr(), res_wrapper_ct4.get_ptr());
// cublasDrotg-NEXT:   return 0;
// cublasDrotg-NEXT:   }();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCtpsv | FileCheck %s -check-prefix=cublasCtpsv
// cublasCtpsv: CUDA API:
// cublasCtpsv-NEXT:   cublasCtpsv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCtpsv-NEXT:               trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasCtpsv-NEXT:               n /*int*/, a /*const cuComplex **/, x /*cuComplex **/,
// cublasCtpsv-NEXT:               incx /*int*/);
// cublasCtpsv-NEXT: Is migrated to:
// cublasCtpsv-NEXT:   oneapi::mkl::blas::column_major::tpsv(handle->get_queue(), upper_lower, trans, unit_nonunit, n, (std::complex<float>*)a, (std::complex<float>*)x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasAxpyEx | FileCheck %s -check-prefix=cublasAxpyEx
// cublasAxpyEx: CUDA API:
// cublasAxpyEx-NEXT:   cublasAxpyEx(handle /*cublasHandle_t*/, n /*int*/, alpha /*const void **/,
// cublasAxpyEx-NEXT:                alphatype /*cudaDataType*/, x /*const void **/,
// cublasAxpyEx-NEXT:                xtype /*cudaDataType*/, incx /*int*/, y /*void **/,
// cublasAxpyEx-NEXT:                ytype /*cudaDataType*/, incy /*int*/,
// cublasAxpyEx-NEXT:                computetype /*cudaDataType*/);
// cublasAxpyEx-NEXT: Is migrated to:
// cublasAxpyEx-NEXT:   dpct::blas::axpy(handle, n, alpha, alphatype, x, xtype, incx, y, ytype, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDtpmv | FileCheck %s -check-prefix=cublasDtpmv
// cublasDtpmv: CUDA API:
// cublasDtpmv-NEXT:   cublasDtpmv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDtpmv-NEXT:               trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasDtpmv-NEXT:               n /*int*/, a /*const double **/, x /*double **/, incx /*int*/);
// cublasDtpmv-NEXT: Is migrated to:
// cublasDtpmv-NEXT:   oneapi::mkl::blas::column_major::tpmv(handle->get_queue(), upper_lower, trans, unit_nonunit, n, a, x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZtrmv | FileCheck %s -check-prefix=cublasZtrmv
// cublasZtrmv: CUDA API:
// cublasZtrmv-NEXT:   cublasZtrmv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZtrmv-NEXT:               trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasZtrmv-NEXT:               n /*int*/, a /*const cuDoubleComplex **/, lda /*int*/,
// cublasZtrmv-NEXT:               x /*cuDoubleComplex **/, incx /*int*/);
// cublasZtrmv-NEXT: Is migrated to:
// cublasZtrmv-NEXT:   oneapi::mkl::blas::column_major::trmv(handle->get_queue(), upper_lower, trans, unit_nonunit, n, (std::complex<double>*)a, lda, (std::complex<double>*)x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSdot | FileCheck %s -check-prefix=cublasSdot
// cublasSdot: CUDA API:
// cublasSdot-NEXT:   cublasSdot(handle /*cublasHandle_t*/, n /*int*/, x /*const float **/,
// cublasSdot-NEXT:              incx /*int*/, y /*const float **/, incy /*int*/, res /*float **/);
// cublasSdot-NEXT: Is migrated to:
// cublasSdot-NEXT:   [&]() {
// cublasSdot-NEXT:   dpct::blas::wrapper_float_out res_wrapper_ct6(handle->get_queue(), res);
// cublasSdot-NEXT:   oneapi::mkl::blas::column_major::dot(handle->get_queue(), n, x, incx, y, incy, res_wrapper_ct6.get_ptr());
// cublasSdot-NEXT:   return 0;
// cublasSdot-NEXT:   }();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasGetVector | FileCheck %s -check-prefix=cublasGetVector
// cublasGetVector: CUDA API:
// cublasGetVector-NEXT:   cublasGetVector(n /*int*/, elementsize /*int*/, x /*const void **/,
// cublasGetVector-NEXT:                   incx /*int*/, y /*void **/, incy /*int*/);
// cublasGetVector-NEXT: Is migrated to:
// cublasGetVector-NEXT:   dpct::blas::matrix_mem_copy(y, x, incy, incx, 1, n, elementsize);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDgemv | FileCheck %s -check-prefix=cublasDgemv
// cublasDgemv: CUDA API:
// cublasDgemv-NEXT:   cublasDgemv(handle /*cublasHandle_t*/, trans /*cublasOperation_t*/, m /*int*/,
// cublasDgemv-NEXT:               n /*int*/, alpha /*const double **/, a /*const double **/,
// cublasDgemv-NEXT:               lda /*int*/, x /*const double **/, incx /*int*/,
// cublasDgemv-NEXT:               beta /*const double **/, y /*double **/, incy /*int*/);
// cublasDgemv-NEXT: Is migrated to:
// cublasDgemv-NEXT:   oneapi::mkl::blas::column_major::gemv(handle->get_queue(), trans, m, n, dpct::get_value(alpha, handle->get_queue()), a, lda, x, incx, dpct::get_value(beta, handle->get_queue()), y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasChpr | FileCheck %s -check-prefix=cublasChpr
// cublasChpr: CUDA API:
// cublasChpr-NEXT:   cublasChpr(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasChpr-NEXT:              n /*int*/, alpha /*const float **/, x /*const cuComplex **/,
// cublasChpr-NEXT:              incx /*int*/, a /*cuComplex **/);
// cublasChpr-NEXT: Is migrated to:
// cublasChpr-NEXT:   oneapi::mkl::blas::column_major::hpr(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)x, incx, (std::complex<float>*)a);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZgeru | FileCheck %s -check-prefix=cublasZgeru
// cublasZgeru: CUDA API:
// cublasZgeru-NEXT:   cublasZgeru(handle /*cublasHandle_t*/, m /*int*/, n /*int*/,
// cublasZgeru-NEXT:               alpha /*const cuDoubleComplex **/, x /*const cuDoubleComplex **/,
// cublasZgeru-NEXT:               incx /*int*/, y /*const cuDoubleComplex **/, incy /*int*/,
// cublasZgeru-NEXT:               a /*cuDoubleComplex **/, lda /*int*/);
// cublasZgeru-NEXT: Is migrated to:
// cublasZgeru-NEXT:   oneapi::mkl::blas::column_major::geru(handle->get_queue(), m, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<double>*)x, incx, (std::complex<double>*)y, incy, (std::complex<double>*)a, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCdgmm | FileCheck %s -check-prefix=cublasCdgmm
// cublasCdgmm: CUDA API:
// cublasCdgmm-NEXT:   cublasCdgmm(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasCdgmm-NEXT:               m /*int*/, n /*int*/, a /*const cuComplex **/, lda /*int*/,
// cublasCdgmm-NEXT:               x /*const cuComplex **/, incx /*int*/, c /*cuComplex **/,
// cublasCdgmm-NEXT:               ldc /*int*/);
// cublasCdgmm-NEXT: Is migrated to:
// cublasCdgmm-NEXT:   oneapi::mkl::blas::column_major::dgmm(handle->get_queue(), left_right, m, n, (std::complex<float>*)a, lda, (std::complex<float>*)x, incx, (std::complex<float>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDcopy | FileCheck %s -check-prefix=cublasDcopy
// cublasDcopy: CUDA API:
// cublasDcopy-NEXT:   cublasDcopy(handle /*cublasHandle_t*/, n /*int*/, x /*const double **/,
// cublasDcopy-NEXT:               incx /*int*/, y /*double **/, incy /*int*/);
// cublasDcopy-NEXT: Is migrated to:
// cublasDcopy-NEXT:   oneapi::mkl::blas::column_major::copy(handle->get_queue(), n, x, incx, y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSsyrkx | FileCheck %s -check-prefix=cublasSsyrkx
// cublasSsyrkx: CUDA API:
// cublasSsyrkx-NEXT:   cublasSsyrkx(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasSsyrkx-NEXT:                trans /*cublasOperation_t*/, n /*int*/, k /*int*/,
// cublasSsyrkx-NEXT:                alpha /*const float **/, a /*const float **/, lda /*int*/,
// cublasSsyrkx-NEXT:                b /*const float **/, ldb /*int*/, beta /*const float **/,
// cublasSsyrkx-NEXT:                c /*float **/, ldc /*int*/);
// cublasSsyrkx-NEXT: Is migrated to:
// cublasSsyrkx-NEXT:   dpct::blas::syrk(handle, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDestroy | FileCheck %s -check-prefix=cublasDestroy
// cublasDestroy: CUDA API:
// cublasDestroy-NEXT:   cublasDestroy(handle /*cublasHandle_t*/);
// cublasDestroy-NEXT: Is migrated to:
// cublasDestroy-NEXT:   delete (handle);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasGetAtomicsMode | FileCheck %s -check-prefix=cublasGetAtomicsMode
// cublasGetAtomicsMode: CUDA API:
// cublasGetAtomicsMode-NEXT:   cublasGetAtomicsMode(handle /*cublasHandle_t*/,
// cublasGetAtomicsMode-NEXT:                        atomics /*cublasAtomicsMode_t **/);
// cublasGetAtomicsMode-NEXT: The API is Removed.

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZsyr2k | FileCheck %s -check-prefix=cublasZsyr2k
// cublasZsyr2k: CUDA API:
// cublasZsyr2k-NEXT:   cublasZsyr2k(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZsyr2k-NEXT:                trans /*cublasOperation_t*/, n /*int*/, k /*int*/,
// cublasZsyr2k-NEXT:                alpha /*const cuDoubleComplex **/, a /*const cuDoubleComplex **/,
// cublasZsyr2k-NEXT:                lda /*int*/, b /*const cuDoubleComplex **/, ldb /*int*/,
// cublasZsyr2k-NEXT:                beta /*const cuDoubleComplex **/, c /*cuDoubleComplex **/,
// cublasZsyr2k-NEXT:                ldc /*int*/);
// cublasZsyr2k-NEXT: Is migrated to:
// cublasZsyr2k-NEXT:   oneapi::mkl::blas::column_major::syr2k(handle->get_queue(), upper_lower, trans, n, k, dpct::get_value(alpha, handle->get_queue()), (std::complex<double>*)a, lda, (std::complex<double>*)b, ldb, dpct::get_value(beta, handle->get_queue()), (std::complex<double>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDrotm | FileCheck %s -check-prefix=cublasDrotm
// cublasDrotm: CUDA API:
// cublasDrotm-NEXT:   cublasDrotm(handle /*cublasHandle_t*/, n /*int*/, x /*double **/,
// cublasDrotm-NEXT:               incx /*int*/, y /*double **/, incy /*int*/,
// cublasDrotm-NEXT:               param /*const double **/);
// cublasDrotm-NEXT: Is migrated to:
// cublasDrotm-NEXT:   [&]() {
// cublasDrotm-NEXT:   dpct::blas::wrapper_double_in res_wrapper_ct6(handle->get_queue(), param, 5);
// cublasDrotm-NEXT:   oneapi::mkl::blas::column_major::rotm(handle->get_queue(), n, x, incx, y, incy, res_wrapper_ct6.get_ptr());
// cublasDrotm-NEXT:   return 0;
// cublasDrotm-NEXT:   }();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCdotu | FileCheck %s -check-prefix=cublasCdotu
// cublasCdotu: CUDA API:
// cublasCdotu-NEXT:   cublasCdotu(handle /*cublasHandle_t*/, n /*int*/, x /*const cuComplex **/,
// cublasCdotu-NEXT:               incx /*int*/, y /*const cuComplex **/, incy /*int*/,
// cublasCdotu-NEXT:               res /*cuComplex **/);
// cublasCdotu-NEXT: Is migrated to:
// cublasCdotu-NEXT:   [&]() {
// cublasCdotu-NEXT:   dpct::blas::wrapper_float2_out res_wrapper_ct6(handle->get_queue(), res);
// cublasCdotu-NEXT:   oneapi::mkl::blas::column_major::dotu(handle->get_queue(), n, (std::complex<float>*)x, incx, (std::complex<float>*)y, incy, (std::complex<float>*)res_wrapper_ct6.get_ptr());
// cublasCdotu-NEXT:   return 0;
// cublasCdotu-NEXT:   }();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDspr | FileCheck %s -check-prefix=cublasDspr
// cublasDspr: CUDA API:
// cublasDspr-NEXT:   cublasDspr(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDspr-NEXT:              n /*int*/, alpha /*const double **/, x /*const double **/,
// cublasDspr-NEXT:              incx /*int*/, a /*double **/);
// cublasDspr-NEXT: Is migrated to:
// cublasDspr-NEXT:   oneapi::mkl::blas::column_major::spr(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), x, incx, a);
