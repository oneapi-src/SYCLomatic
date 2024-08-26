// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-11.8
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6, v11.7, v11.8

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCtrsm_64 | FileCheck %s -check-prefix=cublasCtrsm_64
// cublasCtrsm_64: CUDA API:
// cublasCtrsm_64-NEXT:   cublasCtrsm_64(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasCtrsm_64-NEXT:                  upper_lower /*cublasFillMode_t*/, transa /*cublasOperation_t*/,
// cublasCtrsm_64-NEXT:                  unit_diag /*cublasDiagType_t*/, m /*int64_t*/, n /*int64_t*/,
// cublasCtrsm_64-NEXT:                  alpha /*const cuComplex **/, a /*const cuComplex **/,
// cublasCtrsm_64-NEXT:                  lda /*int64_t*/, b /*cuComplex **/, ldb /*int64_t*/);
// cublasCtrsm_64-NEXT: Is migrated to:
// cublasCtrsm_64-NEXT:   oneapi::mkl::blas::column_major::trsm(handle->get_queue(), left_right, upper_lower, transa, unit_diag, m, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)a, lda, (std::complex<float>*)b, ldb);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZtrsm_64 | FileCheck %s -check-prefix=cublasZtrsm_64
// cublasZtrsm_64: CUDA API:
// cublasZtrsm_64-NEXT:   cublasZtrsm_64(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasZtrsm_64-NEXT:                  upper_lower /*cublasFillMode_t*/, transa /*cublasOperation_t*/,
// cublasZtrsm_64-NEXT:                  unit_diag /*cublasDiagType_t*/, m /*int64_t*/, n /*int64_t*/,
// cublasZtrsm_64-NEXT:                  alpha /*const cuDoubleComplex **/,
// cublasZtrsm_64-NEXT:                  a /*const cuDoubleComplex **/, lda /*int64_t*/,
// cublasZtrsm_64-NEXT:                  b /*cuDoubleComplex **/, ldb /*int64_t*/);
// cublasZtrsm_64-NEXT: Is migrated to:
// cublasZtrsm_64-NEXT:   oneapi::mkl::blas::column_major::trsm(handle->get_queue(), left_right, upper_lower, transa, unit_diag, m, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<double>*)a, lda, (std::complex<double>*)b, ldb);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasStrsv_64 | FileCheck %s -check-prefix=cublasStrsv_64
// cublasStrsv_64: CUDA API:
// cublasStrsv_64-NEXT:   cublasStrsv_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasStrsv_64-NEXT:                  trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasStrsv_64-NEXT:                  n /*int64_t*/, a /*const float **/, lda /*int64_t*/,
// cublasStrsv_64-NEXT:                  x /*float **/, incx /*int64_t*/);
// cublasStrsv_64-NEXT: Is migrated to:
// cublasStrsv_64-NEXT:   oneapi::mkl::blas::column_major::trsv(handle->get_queue(), upper_lower, trans, unit_nonunit, n, a, lda, x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDtrsv_64 | FileCheck %s -check-prefix=cublasDtrsv_64
// cublasDtrsv_64: CUDA API:
// cublasDtrsv_64-NEXT:   cublasDtrsv_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDtrsv_64-NEXT:                  trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasDtrsv_64-NEXT:                  n /*int64_t*/, a /*const double **/, lda /*int64_t*/,
// cublasDtrsv_64-NEXT:                  x /*double **/, incx /*int64_t*/);
// cublasDtrsv_64-NEXT: Is migrated to:
// cublasDtrsv_64-NEXT:   oneapi::mkl::blas::column_major::trsv(handle->get_queue(), upper_lower, trans, unit_nonunit, n, a, lda, x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCtrsv_64 | FileCheck %s -check-prefix=cublasCtrsv_64
// cublasCtrsv_64: CUDA API:
// cublasCtrsv_64-NEXT:   cublasCtrsv_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCtrsv_64-NEXT:                  trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasCtrsv_64-NEXT:                  n /*int64_t*/, a /*const cuComplex **/, lda /*int64_t*/,
// cublasCtrsv_64-NEXT:                  x /*cuComplex **/, incx /*int64_t*/);
// cublasCtrsv_64-NEXT: Is migrated to:
// cublasCtrsv_64-NEXT:   oneapi::mkl::blas::column_major::trsv(handle->get_queue(), upper_lower, trans, unit_nonunit, n, (std::complex<float>*)a, lda, (std::complex<float>*)x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZtrsv_64 | FileCheck %s -check-prefix=cublasZtrsv_64
// cublasZtrsv_64: CUDA API:
// cublasZtrsv_64-NEXT:   cublasZtrsv_64(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZtrsv_64-NEXT:                  trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasZtrsv_64-NEXT:                  n /*int64_t*/, a /*const cuDoubleComplex **/, lda /*int64_t*/,
// cublasZtrsv_64-NEXT:                  x /*cuDoubleComplex **/, incx /*int64_t*/);
// cublasZtrsv_64-NEXT: Is migrated to:
// cublasZtrsv_64-NEXT:   oneapi::mkl::blas::column_major::trsv(handle->get_queue(), upper_lower, trans, unit_nonunit, n, (std::complex<double>*)a, lda, (std::complex<double>*)x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSasum_64 | FileCheck %s -check-prefix=cublasSasum_64
// cublasSasum_64: CUDA API:
// cublasSasum_64-NEXT:   cublasSasum_64(handle /*cublasHandle_t*/, n /*int64_t*/, x /*const float **/,
// cublasSasum_64-NEXT:                  incx /*int64_t*/, res /*float **/);
// cublasSasum_64-NEXT: Is migrated to:
// cublasSasum_64-NEXT:   [&]() {
// cublasSasum_64-NEXT:   dpct::blas::wrapper_float_out res_wrapper_ct4(handle->get_queue(), res);
// cublasSasum_64-NEXT:   oneapi::mkl::blas::column_major::asum(handle->get_queue(), n, x, incx, res_wrapper_ct4.get_ptr());
// cublasSasum_64-NEXT:   return 0;
// cublasSasum_64-NEXT:   }();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDasum_64 | FileCheck %s -check-prefix=cublasDasum_64
// cublasDasum_64: CUDA API:
// cublasDasum_64-NEXT:   cublasDasum_64(handle /*cublasHandle_t*/, n /*int64_t*/, x /*const double **/,
// cublasDasum_64-NEXT:                  incx /*int64_t*/, res /*double **/);
// cublasDasum_64-NEXT: Is migrated to:
// cublasDasum_64-NEXT:   [&]() {
// cublasDasum_64-NEXT:   dpct::blas::wrapper_double_out res_wrapper_ct4(handle->get_queue(), res);
// cublasDasum_64-NEXT:   oneapi::mkl::blas::column_major::asum(handle->get_queue(), n, x, incx, res_wrapper_ct4.get_ptr());
// cublasDasum_64-NEXT:   return 0;
// cublasDasum_64-NEXT:   }();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasScasum_64 | FileCheck %s -check-prefix=cublasScasum_64
// cublasScasum_64: CUDA API:
// cublasScasum_64-NEXT:   cublasScasum_64(handle /*cublasHandle_t*/, n /*int64_t*/,
// cublasScasum_64-NEXT:                   x /*const cuComplex **/, incx /*int64_t*/, res /*float **/);
// cublasScasum_64-NEXT: Is migrated to:
// cublasScasum_64-NEXT:   [&]() {
// cublasScasum_64-NEXT:   dpct::blas::wrapper_float_out res_wrapper_ct4(handle->get_queue(), res);
// cublasScasum_64-NEXT:   oneapi::mkl::blas::column_major::asum(handle->get_queue(), n, (std::complex<float>*)x, incx, res_wrapper_ct4.get_ptr());
// cublasScasum_64-NEXT:   return 0;
// cublasScasum_64-NEXT:   }();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDzasum_64 | FileCheck %s -check-prefix=cublasDzasum_64
// cublasDzasum_64: CUDA API:
// cublasDzasum_64-NEXT:   cublasDzasum_64(handle /*cublasHandle_t*/, n /*int64_t*/,
// cublasDzasum_64-NEXT:                   x /*const cuDoubleComplex **/, incx /*int64_t*/,
// cublasDzasum_64-NEXT:                   res /*double **/);
// cublasDzasum_64-NEXT: Is migrated to:
// cublasDzasum_64-NEXT:   [&]() {
// cublasDzasum_64-NEXT:   dpct::blas::wrapper_double_out res_wrapper_ct4(handle->get_queue(), res);
// cublasDzasum_64-NEXT:   oneapi::mkl::blas::column_major::asum(handle->get_queue(), n, (std::complex<double>*)x, incx, res_wrapper_ct4.get_ptr());
// cublasDzasum_64-NEXT:   return 0;
// cublasDzasum_64-NEXT:   }();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSnrm2_64 | FileCheck %s -check-prefix=cublasSnrm2_64
// cublasSnrm2_64: CUDA API:
// cublasSnrm2_64-NEXT:   cublasSnrm2_64(handle /*cublasHandle_t*/, n /*int64_t*/, x /*const float **/,
// cublasSnrm2_64-NEXT:                  incx /*int64_t*/, res /*float **/);
// cublasSnrm2_64-NEXT: Is migrated to:
// cublasSnrm2_64-NEXT:   [&]() {
// cublasSnrm2_64-NEXT:   dpct::blas::wrapper_float_out res_wrapper_ct4(handle->get_queue(), res);
// cublasSnrm2_64-NEXT:   oneapi::mkl::blas::column_major::nrm2(handle->get_queue(), n, x, incx, res_wrapper_ct4.get_ptr());
// cublasSnrm2_64-NEXT:   return 0;
// cublasSnrm2_64-NEXT:   }();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDnrm2_64 | FileCheck %s -check-prefix=cublasDnrm2_64
// cublasDnrm2_64: CUDA API:
// cublasDnrm2_64-NEXT:   cublasDnrm2_64(handle /*cublasHandle_t*/, n /*int64_t*/, x /*const double **/,
// cublasDnrm2_64-NEXT:                  incx /*int64_t*/, res /*double **/);
// cublasDnrm2_64-NEXT: Is migrated to:
// cublasDnrm2_64-NEXT:   [&]() {
// cublasDnrm2_64-NEXT:   dpct::blas::wrapper_double_out res_wrapper_ct4(handle->get_queue(), res);
// cublasDnrm2_64-NEXT:   oneapi::mkl::blas::column_major::nrm2(handle->get_queue(), n, x, incx, res_wrapper_ct4.get_ptr());
// cublasDnrm2_64-NEXT:   return 0;
// cublasDnrm2_64-NEXT:   }();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasScnrm2_64 | FileCheck %s -check-prefix=cublasScnrm2_64
// cublasScnrm2_64: CUDA API:
// cublasScnrm2_64-NEXT:   cublasScnrm2_64(handle /*cublasHandle_t*/, n /*int64_t*/,
// cublasScnrm2_64-NEXT:                   x /*const cuComplex **/, incx /*int64_t*/, res /*float **/);
// cublasScnrm2_64-NEXT: Is migrated to:
// cublasScnrm2_64-NEXT:   [&]() {
// cublasScnrm2_64-NEXT:   dpct::blas::wrapper_float_out res_wrapper_ct4(handle->get_queue(), res);
// cublasScnrm2_64-NEXT:   oneapi::mkl::blas::column_major::nrm2(handle->get_queue(), n, (std::complex<float>*)x, incx, res_wrapper_ct4.get_ptr());
// cublasScnrm2_64-NEXT:   return 0;
// cublasScnrm2_64-NEXT:   }();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDznrm2_64 | FileCheck %s -check-prefix=cublasDznrm2_64
// cublasDznrm2_64: CUDA API:
// cublasDznrm2_64-NEXT:   cublasDznrm2_64(handle /*cublasHandle_t*/, n /*int64_t*/,
// cublasDznrm2_64-NEXT:                   x /*const cuDoubleComplex **/, incx /*int64_t*/,
// cublasDznrm2_64-NEXT:                   res /*double **/);
// cublasDznrm2_64-NEXT: Is migrated to:
// cublasDznrm2_64-NEXT:   [&]() {
// cublasDznrm2_64-NEXT:   dpct::blas::wrapper_double_out res_wrapper_ct4(handle->get_queue(), res);
// cublasDznrm2_64-NEXT:   oneapi::mkl::blas::column_major::nrm2(handle->get_queue(), n, (std::complex<double>*)x, incx, res_wrapper_ct4.get_ptr());
// cublasDznrm2_64-NEXT:   return 0;
// cublasDznrm2_64-NEXT:   }();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSrotm_64 | FileCheck %s -check-prefix=cublasSrotm_64
// cublasSrotm_64: CUDA API:
// cublasSrotm_64-NEXT:   cublasSrotm_64(handle /*cublasHandle_t*/, n /*int64_t*/, x /*float **/,
// cublasSrotm_64-NEXT:                  incx /*int64_t*/, y /*float **/, incy /*int64_t*/,
// cublasSrotm_64-NEXT:                  param /*const float **/);
// cublasSrotm_64-NEXT: Is migrated to:
// cublasSrotm_64-NEXT:   [&]() {
// cublasSrotm_64-NEXT:   dpct::blas::wrapper_float_in res_wrapper_ct6(handle->get_queue(), param, 5);
// cublasSrotm_64-NEXT:   oneapi::mkl::blas::column_major::rotm(handle->get_queue(), n, x, incx, y, incy, res_wrapper_ct6.get_ptr());
// cublasSrotm_64-NEXT:   return 0;
// cublasSrotm_64-NEXT:   }();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDrotm_64 | FileCheck %s -check-prefix=cublasDrotm_64
// cublasDrotm_64: CUDA API:
// cublasDrotm_64-NEXT:   cublasDrotm_64(handle /*cublasHandle_t*/, n /*int64_t*/, x /*double **/,
// cublasDrotm_64-NEXT:                  incx /*int64_t*/, y /*double **/, incy /*int64_t*/,
// cublasDrotm_64-NEXT:                  param /*const double **/);
// cublasDrotm_64-NEXT: Is migrated to:
// cublasDrotm_64-NEXT:   [&]() {
// cublasDrotm_64-NEXT:   dpct::blas::wrapper_double_in res_wrapper_ct6(handle->get_queue(), param, 5);
// cublasDrotm_64-NEXT:   oneapi::mkl::blas::column_major::rotm(handle->get_queue(), n, x, incx, y, incy, res_wrapper_ct6.get_ptr());
// cublasDrotm_64-NEXT:   return 0;
// cublasDrotm_64-NEXT:   }();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasGetMatrixAsync_64 | FileCheck %s -check-prefix=cublasGetMatrixAsync_64
// cublasGetMatrixAsync_64: CUDA API:
// cublasGetMatrixAsync_64-NEXT:   cublasGetMatrixAsync_64(rows /*int64_t*/, cols /*int64_t*/,
// cublasGetMatrixAsync_64-NEXT:                           elementsize /*int64_t*/, a /*const void **/,
// cublasGetMatrixAsync_64-NEXT:                           lda /*int64_t*/, b /*void **/, ldb /*int64_t*/,
// cublasGetMatrixAsync_64-NEXT:                           stream /*cudaStream_t*/);
// cublasGetMatrixAsync_64-NEXT: Is migrated to:
// cublasGetMatrixAsync_64-NEXT:   dpct::blas::matrix_mem_copy(b, a, ldb, lda, rows, cols, elementsize, dpct::cs::memcpy_direction::automatic, *stream, true);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSetMatrixAsync_64 | FileCheck %s -check-prefix=cublasSetMatrixAsync_64
// cublasSetMatrixAsync_64: CUDA API:
// cublasSetMatrixAsync_64-NEXT:   cublasSetMatrixAsync_64(rows /*int64_t*/, cols /*int64_t*/,
// cublasSetMatrixAsync_64-NEXT:                           elementsize /*int64_t*/, a /*const void **/,
// cublasSetMatrixAsync_64-NEXT:                           lda /*int64_t*/, b /*void **/, ldb /*int64_t*/,
// cublasSetMatrixAsync_64-NEXT:                           stream /*cudaStream_t*/);
// cublasSetMatrixAsync_64-NEXT: Is migrated to:
// cublasSetMatrixAsync_64-NEXT:   dpct::blas::matrix_mem_copy(b, a, ldb, lda, rows, cols, elementsize, dpct::cs::memcpy_direction::automatic, *stream, true);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasGetMatrix_64 | FileCheck %s -check-prefix=cublasGetMatrix_64
// cublasGetMatrix_64: CUDA API:
// cublasGetMatrix_64-NEXT:   cublasGetMatrix_64(rows /*int64_t*/, cols /*int64_t*/,
// cublasGetMatrix_64-NEXT:                      elementsize /*int64_t*/, a /*const void **/,
// cublasGetMatrix_64-NEXT:                      lda /*int64_t*/, b /*void **/, ldb /*int64_t*/);
// cublasGetMatrix_64-NEXT: Is migrated to:
// cublasGetMatrix_64-NEXT:   dpct::blas::matrix_mem_copy(b, a, ldb, lda, rows, cols, elementsize);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSetMatrix_64 | FileCheck %s -check-prefix=cublasSetMatrix_64
// cublasSetMatrix_64: CUDA API:
// cublasSetMatrix_64-NEXT:   cublasSetMatrix_64(rows /*int64_t*/, cols /*int64_t*/,
// cublasSetMatrix_64-NEXT:                      elementsize /*int64_t*/, a /*const void **/,
// cublasSetMatrix_64-NEXT:                      lda /*int64_t*/, b /*void **/, ldb /*int64_t*/);
// cublasSetMatrix_64-NEXT: Is migrated to:
// cublasSetMatrix_64-NEXT:   dpct::blas::matrix_mem_copy(b, a, ldb, lda, rows, cols, elementsize);
