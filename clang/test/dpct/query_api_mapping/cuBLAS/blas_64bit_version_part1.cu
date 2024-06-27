// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-11.8
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6, v11.7, v11.8

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSaxpy_64 | FileCheck %s -check-prefix=cublasSaxpy_64
// cublasSaxpy_64: CUDA API:
// cublasSaxpy_64-NEXT:   cublasSaxpy_64(handle /*cublasHandle_t*/, n /*int64_t*/,
// cublasSaxpy_64-NEXT:                  alpha /*const float **/, x /*const float **/, incx /*int64_t*/,
// cublasSaxpy_64-NEXT:                  y /*float **/, incy /*int64_t*/);
// cublasSaxpy_64-NEXT: Is migrated to:
// cublasSaxpy_64-NEXT:   oneapi::mkl::blas::column_major::axpy(handle->get_queue(), n, dpct::get_value(alpha, handle->get_queue()), x, incx, y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDaxpy_64 | FileCheck %s -check-prefix=cublasDaxpy_64
// cublasDaxpy_64: CUDA API:
// cublasDaxpy_64-NEXT:   cublasDaxpy_64(handle /*cublasHandle_t*/, n /*int64_t*/,
// cublasDaxpy_64-NEXT:                  alpha /*const double **/, x /*const double **/,
// cublasDaxpy_64-NEXT:                  incx /*int64_t*/, y /*double **/, incy /*int64_t*/);
// cublasDaxpy_64-NEXT: Is migrated to:
// cublasDaxpy_64-NEXT:   oneapi::mkl::blas::column_major::axpy(handle->get_queue(), n, dpct::get_value(alpha, handle->get_queue()), x, incx, y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCaxpy_64 | FileCheck %s -check-prefix=cublasCaxpy_64
// cublasCaxpy_64: CUDA API:
// cublasCaxpy_64-NEXT:   cublasCaxpy_64(handle /*cublasHandle_t*/, n /*int64_t*/,
// cublasCaxpy_64-NEXT:                  alpha /*const cuComplex **/, x /*const cuComplex **/,
// cublasCaxpy_64-NEXT:                  incx /*int64_t*/, y /*cuComplex **/, incy /*int64_t*/);
// cublasCaxpy_64-NEXT: Is migrated to:
// cublasCaxpy_64-NEXT:   oneapi::mkl::blas::column_major::axpy(handle->get_queue(), n, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)x, incx, (std::complex<float>*)y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZaxpy_64 | FileCheck %s -check-prefix=cublasZaxpy_64
// cublasZaxpy_64: CUDA API:
// cublasZaxpy_64-NEXT:   cublasZaxpy_64(handle /*cublasHandle_t*/, n /*int64_t*/,
// cublasZaxpy_64-NEXT:                  alpha /*const cuDoubleComplex **/,
// cublasZaxpy_64-NEXT:                  x /*const cuDoubleComplex **/, incx /*int64_t*/,
// cublasZaxpy_64-NEXT:                  y /*cuDoubleComplex **/, incy /*int64_t*/);
// cublasZaxpy_64-NEXT: Is migrated to:
// cublasZaxpy_64-NEXT:   oneapi::mkl::blas::column_major::axpy(handle->get_queue(), n, dpct::get_value(alpha, handle->get_queue()), (std::complex<double>*)x, incx, (std::complex<double>*)y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasScopy_64 | FileCheck %s -check-prefix=cublasScopy_64
// cublasScopy_64: CUDA API:
// cublasScopy_64-NEXT:   cublasScopy_64(handle /*cublasHandle_t*/, n /*int64_t*/, x /*const float **/,
// cublasScopy_64-NEXT:                  incx /*int64_t*/, y /*float **/, incy /*int64_t*/);
// cublasScopy_64-NEXT: Is migrated to:
// cublasScopy_64-NEXT:   oneapi::mkl::blas::column_major::copy(handle->get_queue(), n, x, incx, y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDcopy_64 | FileCheck %s -check-prefix=cublasDcopy_64
// cublasDcopy_64: CUDA API:
// cublasDcopy_64-NEXT:   cublasDcopy_64(handle /*cublasHandle_t*/, n /*int64_t*/, x /*const double **/,
// cublasDcopy_64-NEXT:                  incx /*int64_t*/, y /*double **/, incy /*int64_t*/);
// cublasDcopy_64-NEXT: Is migrated to:
// cublasDcopy_64-NEXT:   oneapi::mkl::blas::column_major::copy(handle->get_queue(), n, x, incx, y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCcopy_64 | FileCheck %s -check-prefix=cublasCcopy_64
// cublasCcopy_64: CUDA API:
// cublasCcopy_64-NEXT:   cublasCcopy_64(handle /*cublasHandle_t*/, n /*int64_t*/,
// cublasCcopy_64-NEXT:                  x /*const cuComplex **/, incx /*int64_t*/, y /*cuComplex **/,
// cublasCcopy_64-NEXT:                  incy /*int64_t*/);
// cublasCcopy_64-NEXT: Is migrated to:
// cublasCcopy_64-NEXT:   oneapi::mkl::blas::column_major::copy(handle->get_queue(), n, (std::complex<float>*)x, incx, (std::complex<float>*)y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZcopy_64 | FileCheck %s -check-prefix=cublasZcopy_64
// cublasZcopy_64: CUDA API:
// cublasZcopy_64-NEXT:   cublasZcopy_64(handle /*cublasHandle_t*/, n /*int64_t*/,
// cublasZcopy_64-NEXT:                  x /*const cuDoubleComplex **/, incx /*int64_t*/,
// cublasZcopy_64-NEXT:                  y /*cuDoubleComplex **/, incy /*int64_t*/);
// cublasZcopy_64-NEXT: Is migrated to:
// cublasZcopy_64-NEXT:   oneapi::mkl::blas::column_major::copy(handle->get_queue(), n, (std::complex<double>*)x, incx, (std::complex<double>*)y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSdgmm_64 | FileCheck %s -check-prefix=cublasSdgmm_64
// cublasSdgmm_64: CUDA API:
// cublasSdgmm_64-NEXT:   cublasSdgmm_64(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasSdgmm_64-NEXT:                  m /*int64_t*/, n /*int64_t*/, a /*const float **/,
// cublasSdgmm_64-NEXT:                  lda /*int64_t*/, x /*const float **/, incx /*int64_t*/,
// cublasSdgmm_64-NEXT:                  c /*float **/, ldc /*int64_t*/);
// cublasSdgmm_64-NEXT: Is migrated to:
// cublasSdgmm_64-NEXT:   oneapi::mkl::blas::column_major::dgmm(handle->get_queue(), left_right, m, n, a, lda, x, incx, c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDdgmm_64 | FileCheck %s -check-prefix=cublasDdgmm_64
// cublasDdgmm_64: CUDA API:
// cublasDdgmm_64-NEXT:   cublasDdgmm_64(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasDdgmm_64-NEXT:                  m /*int64_t*/, n /*int64_t*/, a /*const double **/,
// cublasDdgmm_64-NEXT:                  lda /*int64_t*/, x /*const double **/, incx /*int64_t*/,
// cublasDdgmm_64-NEXT:                  c /*double **/, ldc /*int64_t*/);
// cublasDdgmm_64-NEXT: Is migrated to:
// cublasDdgmm_64-NEXT:   oneapi::mkl::blas::column_major::dgmm(handle->get_queue(), left_right, m, n, a, lda, x, incx, c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCdgmm_64 | FileCheck %s -check-prefix=cublasCdgmm_64
// cublasCdgmm_64: CUDA API:
// cublasCdgmm_64-NEXT:   cublasCdgmm_64(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasCdgmm_64-NEXT:                  m /*int64_t*/, n /*int64_t*/, a /*const cuComplex **/,
// cublasCdgmm_64-NEXT:                  lda /*int64_t*/, x /*const cuComplex **/, incx /*int64_t*/,
// cublasCdgmm_64-NEXT:                  c /*cuComplex **/, ldc /*int64_t*/);
// cublasCdgmm_64-NEXT: Is migrated to:
// cublasCdgmm_64-NEXT:   oneapi::mkl::blas::column_major::dgmm(handle->get_queue(), left_right, m, n, (std::complex<float>*)a, lda, (std::complex<float>*)x, incx, (std::complex<float>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZdgmm_64 | FileCheck %s -check-prefix=cublasZdgmm_64
// cublasZdgmm_64: CUDA API:
// cublasZdgmm_64-NEXT:   cublasZdgmm_64(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasZdgmm_64-NEXT:                  m /*int64_t*/, n /*int64_t*/, a /*const cuDoubleComplex **/,
// cublasZdgmm_64-NEXT:                  lda /*int64_t*/, x /*const cuDoubleComplex **/,
// cublasZdgmm_64-NEXT:                  incx /*int64_t*/, c /*cuDoubleComplex **/, ldc /*int64_t*/);
// cublasZdgmm_64-NEXT: Is migrated to:
// cublasZdgmm_64-NEXT:   oneapi::mkl::blas::column_major::dgmm(handle->get_queue(), left_right, m, n, (std::complex<double>*)a, lda, (std::complex<double>*)x, incx, (std::complex<double>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSdot_64 | FileCheck %s -check-prefix=cublasSdot_64
// cublasSdot_64: CUDA API:
// cublasSdot_64-NEXT:   cublasSdot_64(handle /*cublasHandle_t*/, n /*int64_t*/, x /*const float **/,
// cublasSdot_64-NEXT:                 incx /*int64_t*/, y /*const float **/, incy /*int64_t*/,
// cublasSdot_64-NEXT:                 res /*float **/);
// cublasSdot_64-NEXT: Is migrated to:
// cublasSdot_64-NEXT:   [&]() {
// cublasSdot_64-NEXT:   dpct::blas::wrapper_float_out res_wrapper_ct6(handle->get_queue(), res);
// cublasSdot_64-NEXT:   oneapi::mkl::blas::column_major::dot(handle->get_queue(), n, x, incx, y, incy, res_wrapper_ct6.get_ptr());
// cublasSdot_64-NEXT:   return 0;
// cublasSdot_64-NEXT:   }();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDdot_64 | FileCheck %s -check-prefix=cublasDdot_64
// cublasDdot_64: CUDA API:
// cublasDdot_64-NEXT:   cublasDdot_64(handle /*cublasHandle_t*/, n /*int64_t*/, x /*const double **/,
// cublasDdot_64-NEXT:                 incx /*int64_t*/, y /*const double **/, incy /*int64_t*/,
// cublasDdot_64-NEXT:                 res /*double **/);
// cublasDdot_64-NEXT: Is migrated to:
// cublasDdot_64-NEXT:   [&]() {
// cublasDdot_64-NEXT:   dpct::blas::wrapper_double_out res_wrapper_ct6(handle->get_queue(), res);
// cublasDdot_64-NEXT:   oneapi::mkl::blas::column_major::dot(handle->get_queue(), n, x, incx, y, incy, res_wrapper_ct6.get_ptr());
// cublasDdot_64-NEXT:   return 0;
// cublasDdot_64-NEXT:   }();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCdotc_64 | FileCheck %s -check-prefix=cublasCdotc_64
// cublasCdotc_64: CUDA API:
// cublasCdotc_64-NEXT:   cublasCdotc_64(handle /*cublasHandle_t*/, n /*int64_t*/,
// cublasCdotc_64-NEXT:                  x /*const cuComplex **/, incx /*int64_t*/,
// cublasCdotc_64-NEXT:                  y /*const cuComplex **/, incy /*int64_t*/,
// cublasCdotc_64-NEXT:                  res /*cuComplex **/);
// cublasCdotc_64-NEXT: Is migrated to:
// cublasCdotc_64-NEXT:   [&]() {
// cublasCdotc_64-NEXT:   dpct::blas::wrapper_float2_out res_wrapper_ct6(handle->get_queue(), res);
// cublasCdotc_64-NEXT:   oneapi::mkl::blas::column_major::dotc(handle->get_queue(), n, (std::complex<float>*)x, incx, (std::complex<float>*)y, incy, (std::complex<float>*)res_wrapper_ct6.get_ptr());
// cublasCdotc_64-NEXT:   return 0;
// cublasCdotc_64-NEXT:   }();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZdotc_64 | FileCheck %s -check-prefix=cublasZdotc_64
// cublasZdotc_64: CUDA API:
// cublasZdotc_64-NEXT:   cublasZdotc_64(handle /*cublasHandle_t*/, n /*int64_t*/,
// cublasZdotc_64-NEXT:                  x /*const cuDoubleComplex **/, incx /*int64_t*/,
// cublasZdotc_64-NEXT:                  y /*const cuDoubleComplex **/, incy /*int64_t*/,
// cublasZdotc_64-NEXT:                  res /*cuDoubleComplex **/);
// cublasZdotc_64-NEXT: Is migrated to:
// cublasZdotc_64-NEXT:   [&]() {
// cublasZdotc_64-NEXT:   dpct::blas::wrapper_double2_out res_wrapper_ct6(handle->get_queue(), res);
// cublasZdotc_64-NEXT:   oneapi::mkl::blas::column_major::dotc(handle->get_queue(), n, (std::complex<double>*)x, incx, (std::complex<double>*)y, incy, (std::complex<double>*)res_wrapper_ct6.get_ptr());
// cublasZdotc_64-NEXT:   return 0;
// cublasZdotc_64-NEXT:   }();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCdotu_64 | FileCheck %s -check-prefix=cublasCdotu_64
// cublasCdotu_64: CUDA API:
// cublasCdotu_64-NEXT:   cublasCdotu_64(handle /*cublasHandle_t*/, n /*int64_t*/,
// cublasCdotu_64-NEXT:                  x /*const cuComplex **/, incx /*int64_t*/,
// cublasCdotu_64-NEXT:                  y /*const cuComplex **/, incy /*int64_t*/,
// cublasCdotu_64-NEXT:                  res /*cuComplex **/);
// cublasCdotu_64-NEXT: Is migrated to:
// cublasCdotu_64-NEXT:   [&]() {
// cublasCdotu_64-NEXT:   dpct::blas::wrapper_float2_out res_wrapper_ct6(handle->get_queue(), res);
// cublasCdotu_64-NEXT:   oneapi::mkl::blas::column_major::dotu(handle->get_queue(), n, (std::complex<float>*)x, incx, (std::complex<float>*)y, incy, (std::complex<float>*)res_wrapper_ct6.get_ptr());
// cublasCdotu_64-NEXT:   return 0;
// cublasCdotu_64-NEXT:   }();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZdotu_64 | FileCheck %s -check-prefix=cublasZdotu_64
// cublasZdotu_64: CUDA API:
// cublasZdotu_64-NEXT:   cublasZdotu_64(handle /*cublasHandle_t*/, n /*int64_t*/,
// cublasZdotu_64-NEXT:                  x /*const cuDoubleComplex **/, incx /*int64_t*/,
// cublasZdotu_64-NEXT:                  y /*const cuDoubleComplex **/, incy /*int64_t*/,
// cublasZdotu_64-NEXT:                  res /*cuDoubleComplex **/);
// cublasZdotu_64-NEXT: Is migrated to:
// cublasZdotu_64-NEXT:   [&]() {
// cublasZdotu_64-NEXT:   dpct::blas::wrapper_double2_out res_wrapper_ct6(handle->get_queue(), res);
// cublasZdotu_64-NEXT:   oneapi::mkl::blas::column_major::dotu(handle->get_queue(), n, (std::complex<double>*)x, incx, (std::complex<double>*)y, incy, (std::complex<double>*)res_wrapper_ct6.get_ptr());
// cublasZdotu_64-NEXT:   return 0;
// cublasZdotu_64-NEXT:   }();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSgbmv_64 | FileCheck %s -check-prefix=cublasSgbmv_64
// cublasSgbmv_64: CUDA API:
// cublasSgbmv_64-NEXT:   cublasSgbmv_64(handle /*cublasHandle_t*/, trans /*cublasOperation_t*/,
// cublasSgbmv_64-NEXT:                  m /*int64_t*/, n /*int64_t*/, kl /*int64_t*/, ku /*int64_t*/,
// cublasSgbmv_64-NEXT:                  alpha /*const float **/, a /*const float **/, lda /*int64_t*/,
// cublasSgbmv_64-NEXT:                  x /*const float **/, incx /*int64_t*/, beta /*const float **/,
// cublasSgbmv_64-NEXT:                  y /*float **/, incy /*int64_t*/);
// cublasSgbmv_64-NEXT: Is migrated to:
// cublasSgbmv_64-NEXT:   oneapi::mkl::blas::column_major::gbmv(handle->get_queue(), trans, m, n, kl, ku, dpct::get_value(alpha, handle->get_queue()), a, lda, x, incx, dpct::get_value(beta, handle->get_queue()), y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDgbmv_64 | FileCheck %s -check-prefix=cublasDgbmv_64
// cublasDgbmv_64: CUDA API:
// cublasDgbmv_64-NEXT:   cublasDgbmv_64(handle /*cublasHandle_t*/, trans /*cublasOperation_t*/,
// cublasDgbmv_64-NEXT:                  m /*int64_t*/, n /*int64_t*/, kl /*int64_t*/, ku /*int64_t*/,
// cublasDgbmv_64-NEXT:                  alpha /*const double **/, a /*const double **/,
// cublasDgbmv_64-NEXT:                  lda /*int64_t*/, x /*const double **/, incx /*int64_t*/,
// cublasDgbmv_64-NEXT:                  beta /*const double **/, y /*double **/, incy /*int64_t*/);
// cublasDgbmv_64-NEXT: Is migrated to:
// cublasDgbmv_64-NEXT:   oneapi::mkl::blas::column_major::gbmv(handle->get_queue(), trans, m, n, kl, ku, dpct::get_value(alpha, handle->get_queue()), a, lda, x, incx, dpct::get_value(beta, handle->get_queue()), y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCgbmv_64 | FileCheck %s -check-prefix=cublasCgbmv_64
// cublasCgbmv_64: CUDA API:
// cublasCgbmv_64-NEXT:   cublasCgbmv_64(handle /*cublasHandle_t*/, trans /*cublasOperation_t*/,
// cublasCgbmv_64-NEXT:                  m /*int64_t*/, n /*int64_t*/, kl /*int64_t*/, ku /*int64_t*/,
// cublasCgbmv_64-NEXT:                  alpha /*const cuComplex **/, a /*const cuComplex **/,
// cublasCgbmv_64-NEXT:                  lda /*int64_t*/, x /*const cuComplex **/, incx /*int64_t*/,
// cublasCgbmv_64-NEXT:                  beta /*const cuComplex **/, y /*cuComplex **/,
// cublasCgbmv_64-NEXT:                  incy /*int64_t*/);
// cublasCgbmv_64-NEXT: Is migrated to:
// cublasCgbmv_64-NEXT:   oneapi::mkl::blas::column_major::gbmv(handle->get_queue(), trans, m, n, kl, ku, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)a, lda, (std::complex<float>*)x, incx, dpct::get_value(beta, handle->get_queue()), (std::complex<float>*)y, incy);
