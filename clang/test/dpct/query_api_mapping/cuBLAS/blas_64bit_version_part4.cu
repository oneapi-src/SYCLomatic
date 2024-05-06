// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-11.8
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6, v11.7, v11.8

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSsyrk_64 | FileCheck %s -check-prefix=cublasSsyrk_64
// cublasSsyrk_64: CUDA API:
// cublasSsyrk_64-NEXT:   cublasSsyrk_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasSsyrk_64-NEXT:                  trans /*cublasOperation_t*/, n /*int64_t*/, k /*int64_t*/,
// cublasSsyrk_64-NEXT:                  alpha /*const float **/, a /*const float **/, lda /*int64_t*/,
// cublasSsyrk_64-NEXT:                  beta /*const float **/, c /*float **/, ldc /*int64_t*/);
// cublasSsyrk_64-NEXT: Is migrated to:
// cublasSsyrk_64-NEXT:   oneapi::mkl::blas::column_major::syrk(handle->get_queue(), upper_lower, trans, n, k, dpct::get_value(alpha, handle->get_queue()), a, lda, dpct::get_value(beta, handle->get_queue()), c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDsyrk_64 | FileCheck %s -check-prefix=cublasDsyrk_64
// cublasDsyrk_64: CUDA API:
// cublasDsyrk_64-NEXT:   cublasDsyrk_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDsyrk_64-NEXT:                  trans /*cublasOperation_t*/, n /*int64_t*/, k /*int64_t*/,
// cublasDsyrk_64-NEXT:                  alpha /*const double **/, a /*const double **/,
// cublasDsyrk_64-NEXT:                  lda /*int64_t*/, beta /*const double **/, c /*double **/,
// cublasDsyrk_64-NEXT:                  ldc /*int64_t*/);
// cublasDsyrk_64-NEXT: Is migrated to:
// cublasDsyrk_64-NEXT:   oneapi::mkl::blas::column_major::syrk(handle->get_queue(), upper_lower, trans, n, k, dpct::get_value(alpha, handle->get_queue()), a, lda, dpct::get_value(beta, handle->get_queue()), c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCsyrk_64 | FileCheck %s -check-prefix=cublasCsyrk_64
// cublasCsyrk_64: CUDA API:
// cublasCsyrk_64-NEXT:   cublasCsyrk_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCsyrk_64-NEXT:                  trans /*cublasOperation_t*/, n /*int64_t*/, k /*int64_t*/,
// cublasCsyrk_64-NEXT:                  alpha /*const cuComplex **/, a /*const cuComplex **/,
// cublasCsyrk_64-NEXT:                  lda /*int64_t*/, beta /*const cuComplex **/, c /*cuComplex **/,
// cublasCsyrk_64-NEXT:                  ldc /*int64_t*/);
// cublasCsyrk_64-NEXT: Is migrated to:
// cublasCsyrk_64-NEXT:   oneapi::mkl::blas::column_major::syrk(handle->get_queue(), upper_lower, trans, n, k, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)a, lda, dpct::get_value(beta, handle->get_queue()), (std::complex<float>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZsyrk_64 | FileCheck %s -check-prefix=cublasZsyrk_64
// cublasZsyrk_64: CUDA API:
// cublasZsyrk_64-NEXT:   cublasZsyrk_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZsyrk_64-NEXT:                  trans /*cublasOperation_t*/, n /*int64_t*/, k /*int64_t*/,
// cublasZsyrk_64-NEXT:                  alpha /*const cuDoubleComplex **/,
// cublasZsyrk_64-NEXT:                  a /*const cuDoubleComplex **/, lda /*int64_t*/,
// cublasZsyrk_64-NEXT:                  beta /*const cuDoubleComplex **/, c /*cuDoubleComplex **/,
// cublasZsyrk_64-NEXT:                  ldc /*int64_t*/);
// cublasZsyrk_64-NEXT: Is migrated to:
// cublasZsyrk_64-NEXT:   oneapi::mkl::blas::column_major::syrk(handle->get_queue(), upper_lower, trans, n, k, dpct::get_value(alpha, handle->get_queue()), (std::complex<double>*)a, lda, dpct::get_value(beta, handle->get_queue()), (std::complex<double>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasStbmv_64 | FileCheck %s -check-prefix=cublasStbmv_64
// cublasStbmv_64: CUDA API:
// cublasStbmv_64-NEXT:   cublasStbmv_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasStbmv_64-NEXT:                  trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasStbmv_64-NEXT:                  n /*int64_t*/, k /*int64_t*/, a /*const float **/,
// cublasStbmv_64-NEXT:                  lda /*int64_t*/, x /*float **/, incx /*int64_t*/);
// cublasStbmv_64-NEXT: Is migrated to:
// cublasStbmv_64-NEXT:   oneapi::mkl::blas::column_major::tbmv(handle->get_queue(), upper_lower, trans, unit_nonunit, n, k, a, lda, x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDtbmv_64 | FileCheck %s -check-prefix=cublasDtbmv_64
// cublasDtbmv_64: CUDA API:
// cublasDtbmv_64-NEXT:   cublasDtbmv_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDtbmv_64-NEXT:                  trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasDtbmv_64-NEXT:                  n /*int64_t*/, k /*int64_t*/, a /*const double **/,
// cublasDtbmv_64-NEXT:                  lda /*int64_t*/, x /*double **/, incx /*int64_t*/);
// cublasDtbmv_64-NEXT: Is migrated to:
// cublasDtbmv_64-NEXT:   oneapi::mkl::blas::column_major::tbmv(handle->get_queue(), upper_lower, trans, unit_nonunit, n, k, a, lda, x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCtbmv_64 | FileCheck %s -check-prefix=cublasCtbmv_64
// cublasCtbmv_64: CUDA API:
// cublasCtbmv_64-NEXT:   cublasCtbmv_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCtbmv_64-NEXT:                  trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasCtbmv_64-NEXT:                  n /*int64_t*/, k /*int64_t*/, a /*const cuComplex **/,
// cublasCtbmv_64-NEXT:                  lda /*int64_t*/, x /*cuComplex **/, incx /*int64_t*/);
// cublasCtbmv_64-NEXT: Is migrated to:
// cublasCtbmv_64-NEXT:   oneapi::mkl::blas::column_major::tbmv(handle->get_queue(), upper_lower, trans, unit_nonunit, n, k, (std::complex<float>*)a, lda, (std::complex<float>*)x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZtbmv_64 | FileCheck %s -check-prefix=cublasZtbmv_64
// cublasZtbmv_64: CUDA API:
// cublasZtbmv_64-NEXT:   cublasZtbmv_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZtbmv_64-NEXT:                  trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasZtbmv_64-NEXT:                  n /*int64_t*/, k /*int64_t*/, a /*const cuDoubleComplex **/,
// cublasZtbmv_64-NEXT:                  lda /*int64_t*/, x /*cuDoubleComplex **/, incx /*int64_t*/);
// cublasZtbmv_64-NEXT: Is migrated to:
// cublasZtbmv_64-NEXT:   oneapi::mkl::blas::column_major::tbmv(handle->get_queue(), upper_lower, trans, unit_nonunit, n, k, (std::complex<double>*)a, lda, (std::complex<double>*)x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasStpmv_64 | FileCheck %s -check-prefix=cublasStpmv_64
// cublasStpmv_64: CUDA API:
// cublasStpmv_64-NEXT:   cublasStpmv_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasStpmv_64-NEXT:                  trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasStpmv_64-NEXT:                  n /*int64_t*/, a /*const float **/, x /*float **/,
// cublasStpmv_64-NEXT:                  incx /*int64_t*/);
// cublasStpmv_64-NEXT: Is migrated to:
// cublasStpmv_64-NEXT:   oneapi::mkl::blas::column_major::tpmv(handle->get_queue(), upper_lower, trans, unit_nonunit, n, a, x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDtpmv_64 | FileCheck %s -check-prefix=cublasDtpmv_64
// cublasDtpmv_64: CUDA API:
// cublasDtpmv_64-NEXT:   cublasDtpmv_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDtpmv_64-NEXT:                  trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasDtpmv_64-NEXT:                  n /*int64_t*/, a /*const double **/, x /*double **/,
// cublasDtpmv_64-NEXT:                  incx /*int64_t*/);
// cublasDtpmv_64-NEXT: Is migrated to:
// cublasDtpmv_64-NEXT:   oneapi::mkl::blas::column_major::tpmv(handle->get_queue(), upper_lower, trans, unit_nonunit, n, a, x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCtpmv_64 | FileCheck %s -check-prefix=cublasCtpmv_64
// cublasCtpmv_64: CUDA API:
// cublasCtpmv_64-NEXT:   cublasCtpmv_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCtpmv_64-NEXT:                  trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasCtpmv_64-NEXT:                  n /*int64_t*/, a /*const cuComplex **/, x /*cuComplex **/,
// cublasCtpmv_64-NEXT:                  incx /*int64_t*/);
// cublasCtpmv_64-NEXT: Is migrated to:
// cublasCtpmv_64-NEXT:   oneapi::mkl::blas::column_major::tpmv(handle->get_queue(), upper_lower, trans, unit_nonunit, n, (std::complex<float>*)a, (std::complex<float>*)x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZtpmv_64 | FileCheck %s -check-prefix=cublasZtpmv_64
// cublasZtpmv_64: CUDA API:
// cublasZtpmv_64-NEXT:   cublasZtpmv_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZtpmv_64-NEXT:                  trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasZtpmv_64-NEXT:                  n /*int64_t*/, a /*const cuDoubleComplex **/,
// cublasZtpmv_64-NEXT:                  x /*cuDoubleComplex **/, incx /*int64_t*/);
// cublasZtpmv_64-NEXT: Is migrated to:
// cublasZtpmv_64-NEXT:   oneapi::mkl::blas::column_major::tpmv(handle->get_queue(), upper_lower, trans, unit_nonunit, n, (std::complex<double>*)a, (std::complex<double>*)x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasStpsv_64 | FileCheck %s -check-prefix=cublasStpsv_64
// cublasStpsv_64: CUDA API:
// cublasStpsv_64-NEXT:   cublasStpsv_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasStpsv_64-NEXT:                  trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasStpsv_64-NEXT:                  n /*int64_t*/, a /*const float **/, x /*float **/,
// cublasStpsv_64-NEXT:                  incx /*int64_t*/);
// cublasStpsv_64-NEXT: Is migrated to:
// cublasStpsv_64-NEXT:   oneapi::mkl::blas::column_major::tpsv(handle->get_queue(), upper_lower, trans, unit_nonunit, n, a, x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDtpsv_64 | FileCheck %s -check-prefix=cublasDtpsv_64
// cublasDtpsv_64: CUDA API:
// cublasDtpsv_64-NEXT:   cublasDtpsv_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDtpsv_64-NEXT:                  trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasDtpsv_64-NEXT:                  n /*int64_t*/, a /*const double **/, x /*double **/,
// cublasDtpsv_64-NEXT:                  incx /*int64_t*/);
// cublasDtpsv_64-NEXT: Is migrated to:
// cublasDtpsv_64-NEXT:   oneapi::mkl::blas::column_major::tpsv(handle->get_queue(), upper_lower, trans, unit_nonunit, n, a, x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCtpsv_64 | FileCheck %s -check-prefix=cublasCtpsv_64
// cublasCtpsv_64: CUDA API:
// cublasCtpsv_64-NEXT:   cublasCtpsv_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCtpsv_64-NEXT:                  trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasCtpsv_64-NEXT:                  n /*int64_t*/, a /*const cuComplex **/, x /*cuComplex **/,
// cublasCtpsv_64-NEXT:                  incx /*int64_t*/);
// cublasCtpsv_64-NEXT: Is migrated to:
// cublasCtpsv_64-NEXT:   oneapi::mkl::blas::column_major::tpsv(handle->get_queue(), upper_lower, trans, unit_nonunit, n, (std::complex<float>*)a, (std::complex<float>*)x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZtpsv_64 | FileCheck %s -check-prefix=cublasZtpsv_64
// cublasZtpsv_64: CUDA API:
// cublasZtpsv_64-NEXT:   cublasZtpsv_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZtpsv_64-NEXT:                  trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasZtpsv_64-NEXT:                  n /*int64_t*/, a /*const cuDoubleComplex **/,
// cublasZtpsv_64-NEXT:                  x /*cuDoubleComplex **/, incx /*int64_t*/);
// cublasZtpsv_64-NEXT: Is migrated to:
// cublasZtpsv_64-NEXT:   oneapi::mkl::blas::column_major::tpsv(handle->get_queue(), upper_lower, trans, unit_nonunit, n, (std::complex<double>*)a, (std::complex<double>*)x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasStrmv_64 | FileCheck %s -check-prefix=cublasStrmv_64
// cublasStrmv_64: CUDA API:
// cublasStrmv_64-NEXT:   cublasStrmv_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasStrmv_64-NEXT:                  trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasStrmv_64-NEXT:                  n /*int64_t*/, a /*const float **/, lda /*int64_t*/,
// cublasStrmv_64-NEXT:                  x /*float **/, incx /*int64_t*/);
// cublasStrmv_64-NEXT: Is migrated to:
// cublasStrmv_64-NEXT:   oneapi::mkl::blas::column_major::trmv(handle->get_queue(), upper_lower, trans, unit_nonunit, n, a, lda, x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDtrmv_64 | FileCheck %s -check-prefix=cublasDtrmv_64
// cublasDtrmv_64: CUDA API:
// cublasDtrmv_64-NEXT:   cublasDtrmv_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDtrmv_64-NEXT:                  trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasDtrmv_64-NEXT:                  n /*int64_t*/, a /*const double **/, lda /*int64_t*/,
// cublasDtrmv_64-NEXT:                  x /*double **/, incx /*int64_t*/);
// cublasDtrmv_64-NEXT: Is migrated to:
// cublasDtrmv_64-NEXT:   oneapi::mkl::blas::column_major::trmv(handle->get_queue(), upper_lower, trans, unit_nonunit, n, a, lda, x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCtrmv_64 | FileCheck %s -check-prefix=cublasCtrmv_64
// cublasCtrmv_64: CUDA API:
// cublasCtrmv_64-NEXT:   cublasCtrmv_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCtrmv_64-NEXT:                  trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasCtrmv_64-NEXT:                  n /*int64_t*/, a /*const cuComplex **/, lda /*int64_t*/,
// cublasCtrmv_64-NEXT:                  x /*cuComplex **/, incx /*int64_t*/);
// cublasCtrmv_64-NEXT: Is migrated to:
// cublasCtrmv_64-NEXT:   oneapi::mkl::blas::column_major::trmv(handle->get_queue(), upper_lower, trans, unit_nonunit, n, (std::complex<float>*)a, lda, (std::complex<float>*)x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZtrmv_64 | FileCheck %s -check-prefix=cublasZtrmv_64
// cublasZtrmv_64: CUDA API:
// cublasZtrmv_64-NEXT:   cublasZtrmv_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZtrmv_64-NEXT:                  trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasZtrmv_64-NEXT:                  n /*int64_t*/, a /*const cuDoubleComplex **/, lda /*int64_t*/,
// cublasZtrmv_64-NEXT:                  x /*cuDoubleComplex **/, incx /*int64_t*/);
// cublasZtrmv_64-NEXT: Is migrated to:
// cublasZtrmv_64-NEXT:   oneapi::mkl::blas::column_major::trmv(handle->get_queue(), upper_lower, trans, unit_nonunit, n, (std::complex<double>*)a, lda, (std::complex<double>*)x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasStrsm_64 | FileCheck %s -check-prefix=cublasStrsm_64
// cublasStrsm_64: CUDA API:
// cublasStrsm_64-NEXT:   cublasStrsm_64(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasStrsm_64-NEXT:                  upper_lower /*cublasFillMode_t*/, transa /*cublasOperation_t*/,
// cublasStrsm_64-NEXT:                  unit_diag /*cublasDiagType_t*/, m /*int64_t*/, n /*int64_t*/,
// cublasStrsm_64-NEXT:                  alpha /*const float **/, a /*const float **/, lda /*int64_t*/,
// cublasStrsm_64-NEXT:                  b /*float **/, ldb /*int64_t*/);
// cublasStrsm_64-NEXT: Is migrated to:
// cublasStrsm_64-NEXT:   oneapi::mkl::blas::column_major::trsm(handle->get_queue(), left_right, upper_lower, transa, unit_diag, m, n, dpct::get_value(alpha, handle->get_queue()), a, lda, b, ldb);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDtrsm_64 | FileCheck %s -check-prefix=cublasDtrsm_64
// cublasDtrsm_64: CUDA API:
// cublasDtrsm_64-NEXT:   cublasDtrsm_64(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasDtrsm_64-NEXT:                  upper_lower /*cublasFillMode_t*/, transa /*cublasOperation_t*/,
// cublasDtrsm_64-NEXT:                  unit_diag /*cublasDiagType_t*/, m /*int64_t*/, n /*int64_t*/,
// cublasDtrsm_64-NEXT:                  alpha /*const double **/, a /*const double **/,
// cublasDtrsm_64-NEXT:                  lda /*int64_t*/, b /*double **/, ldb /*int64_t*/);
// cublasDtrsm_64-NEXT: Is migrated to:
// cublasDtrsm_64-NEXT:   oneapi::mkl::blas::column_major::trsm(handle->get_queue(), left_right, upper_lower, transa, unit_diag, m, n, dpct::get_value(alpha, handle->get_queue()), a, lda, b, ldb);
