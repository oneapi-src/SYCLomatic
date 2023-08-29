// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZher2k | FileCheck %s -check-prefix=cublasZher2k
// cublasZher2k: CUDA API:
// cublasZher2k-NEXT:   cublasZher2k(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZher2k-NEXT:                trans /*cublasOperation_t*/, n /*int*/, k /*int*/,
// cublasZher2k-NEXT:                alpha /*const cuDoubleComplex **/, a /*const cuDoubleComplex **/,
// cublasZher2k-NEXT:                lda /*int*/, b /*const cuDoubleComplex **/, ldb /*int*/,
// cublasZher2k-NEXT:                beta /*const double **/, c /*cuDoubleComplex **/, ldc /*int*/);
// cublasZher2k-NEXT: Is migrated to:
// cublasZher2k-NEXT:   oneapi::mkl::blas::column_major::her2k(handle->get_queue(), upper_lower, trans, n, k, dpct::get_value(alpha, handle->get_queue()), (std::complex<double>*)a, lda, (std::complex<double>*)b, ldb, dpct::get_value(beta, handle->get_queue()), (std::complex<double>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDsyr2k | FileCheck %s -check-prefix=cublasDsyr2k
// cublasDsyr2k: CUDA API:
// cublasDsyr2k-NEXT:   cublasDsyr2k(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDsyr2k-NEXT:                trans /*cublasOperation_t*/, n /*int*/, k /*int*/,
// cublasDsyr2k-NEXT:                alpha /*const double **/, a /*const double **/, lda /*int*/,
// cublasDsyr2k-NEXT:                b /*const double **/, ldb /*int*/, beta /*const double **/,
// cublasDsyr2k-NEXT:                c /*double **/, ldc /*int*/);
// cublasDsyr2k-NEXT: Is migrated to:
// cublasDsyr2k-NEXT:   oneapi::mkl::blas::column_major::syr2k(handle->get_queue(), upper_lower, trans, n, k, dpct::get_value(alpha, handle->get_queue()), a, lda, b, ldb, dpct::get_value(beta, handle->get_queue()), c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDgemm | FileCheck %s -check-prefix=cublasDgemm
// cublasDgemm: CUDA API:
// cublasDgemm-NEXT:   cublasDgemm(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasDgemm-NEXT:               transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
// cublasDgemm-NEXT:               alpha /*const double **/, a /*const double **/, lda /*int*/,
// cublasDgemm-NEXT:               b /*const double **/, ldb /*int*/, beta /*const double **/,
// cublasDgemm-NEXT:               c /*double **/, ldc /*int*/);
// cublasDgemm-NEXT: Is migrated to:
// cublasDgemm-NEXT:   oneapi::mkl::blas::column_major::gemm(handle->get_queue(), transa, transb, m, n, k, dpct::get_value(alpha, handle->get_queue()), a, lda, b, ldb, dpct::get_value(beta, handle->get_queue()), c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasGetPointerMode | FileCheck %s -check-prefix=cublasGetPointerMode
// cublasGetPointerMode: CUDA API:
// cublasGetPointerMode-NEXT:   cublasGetPointerMode(handle /*cublasHandle_t*/,
// cublasGetPointerMode-NEXT:                        host_device /*cublasPointerMode_t **/);
// cublasGetPointerMode-NEXT: The API is Removed.

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSrotg | FileCheck %s -check-prefix=cublasSrotg
// cublasSrotg: CUDA API:
// cublasSrotg-NEXT:   cublasSrotg(handle /*cublasHandle_t*/, a /*float **/, b /*float **/,
// cublasSrotg-NEXT:               c /*float **/, s /*float **/);
// cublasSrotg-NEXT: Is migrated to:
// cublasSrotg-NEXT:   float* a_ct{{[0-9]+}} = a;
// cublasSrotg-NEXT:   float* b_ct{{[0-9]+}} = b;
// cublasSrotg-NEXT:   float* c_ct{{[0-9]+}} = c;
// cublasSrotg-NEXT:   float* s_ct{{[0-9]+}} = s;
// cublasSrotg-NEXT:   if(sycl::get_pointer_type(a, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(a, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
// cublasSrotg-NEXT:     a_ct{{[0-9]+}} = sycl::malloc_shared<float>(4, dpct::get_default_queue());
// cublasSrotg-NEXT:     b_ct{{[0-9]+}} = a_ct{{[0-9]+}} + 1;
// cublasSrotg-NEXT:     c_ct{{[0-9]+}} = a_ct{{[0-9]+}} + 2;
// cublasSrotg-NEXT:     s_ct{{[0-9]+}} = a_ct{{[0-9]+}} + 3;
// cublasSrotg-NEXT:     *a_ct{{[0-9]+}} = *a;
// cublasSrotg-NEXT:     *b_ct{{[0-9]+}} = *b;
// cublasSrotg-NEXT:     *c_ct{{[0-9]+}} = *c;
// cublasSrotg-NEXT:     *s_ct{{[0-9]+}} = *s;
// cublasSrotg-NEXT:   }
// cublasSrotg-NEXT:   oneapi::mkl::blas::column_major::rotg(handle->get_queue(), a_ct{{[0-9]+}}, b_ct{{[0-9]+}}, c_ct{{[0-9]+}}, s_ct{{[0-9]+}});
// cublasSrotg-NEXT:   if(sycl::get_pointer_type(a, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(a, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
// cublasSrotg-NEXT:     handle->get_queue().wait();
// cublasSrotg-NEXT:     *a = *a_ct{{[0-9]+}};
// cublasSrotg-NEXT:     *b = *b_ct{{[0-9]+}};
// cublasSrotg-NEXT:     *c = *c_ct{{[0-9]+}};
// cublasSrotg-NEXT:     *s = *s_ct{{[0-9]+}};
// cublasSrotg-NEXT:     sycl::free(a_ct{{[0-9]+}}, dpct::get_default_queue());
// cublasSrotg-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSgemmEx | FileCheck %s -check-prefix=cublasSgemmEx
// cublasSgemmEx: CUDA API:
// cublasSgemmEx-NEXT:   cublasSgemmEx(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasSgemmEx-NEXT:                 transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
// cublasSgemmEx-NEXT:                 alpha /*const float **/, a /*const void **/,
// cublasSgemmEx-NEXT:                 atype /*cudaDataType*/, lda /*int*/, b /*const void **/,
// cublasSgemmEx-NEXT:                 btype /*cudaDataType*/, ldb /*int*/, beta /*const float **/,
// cublasSgemmEx-NEXT:                 c /*void **/, ctype /*cudaDataType*/, ldc /*int*/);
// cublasSgemmEx-NEXT: Is migrated to:
// cublasSgemmEx-NEXT:   dpct::gemm(handle->get_queue(), transa, transb, m, n, k, alpha, a, atype, lda, b, btype, ldb, beta, c, ctype, ldc, dpct::library_data_t::real_float);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDrotmg | FileCheck %s -check-prefix=cublasDrotmg
// cublasDrotmg: CUDA API:
// cublasDrotmg-NEXT:   cublasDrotmg(handle /*cublasHandle_t*/, d1 /*double **/, d2 /*double **/,
// cublasDrotmg-NEXT:                x1 /*double **/, y1 /*const double **/, param /*double **/);
// cublasDrotmg-NEXT: Is migrated to:
// cublasDrotmg-NEXT:   double* d1_ct{{[0-9]+}} = d1;
// cublasDrotmg-NEXT:   double* d2_ct{{[0-9]+}} = d2;
// cublasDrotmg-NEXT:   double* x1_ct{{[0-9]+}} = x1;
// cublasDrotmg-NEXT:   double* param_ct{{[0-9]+}} = param;
// cublasDrotmg-NEXT:   if(sycl::get_pointer_type(d1, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(d1, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
// cublasDrotmg-NEXT:     d1_ct{{[0-9]+}} = sycl::malloc_shared<double>(8, dpct::get_default_queue());
// cublasDrotmg-NEXT:     d2_ct{{[0-9]+}} = d1_ct{{[0-9]+}} + 1;
// cublasDrotmg-NEXT:     x1_ct{{[0-9]+}} = d1_ct{{[0-9]+}} + 2;
// cublasDrotmg-NEXT:     param_ct{{[0-9]+}} = d1_ct{{[0-9]+}} + 3;
// cublasDrotmg-NEXT:     *d1_ct{{[0-9]+}} = *d1;
// cublasDrotmg-NEXT:     *d2_ct{{[0-9]+}} = *d2;
// cublasDrotmg-NEXT:     *x1_ct{{[0-9]+}} = *x1;
// cublasDrotmg-NEXT:   }
// cublasDrotmg-NEXT:   oneapi::mkl::blas::column_major::rotmg(handle->get_queue(), d1_ct{{[0-9]+}}, d2_ct{{[0-9]+}}, x1_ct{{[0-9]+}}, dpct::get_value(y1, handle->get_queue()), param_ct{{[0-9]+}});
// cublasDrotmg-NEXT:   if(sycl::get_pointer_type(d1, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(d1, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
// cublasDrotmg-NEXT:     handle->get_queue().wait();
// cublasDrotmg-NEXT:     *d1 = *d1_ct{{[0-9]+}};
// cublasDrotmg-NEXT:     *d2 = *d2_ct{{[0-9]+}};
// cublasDrotmg-NEXT:     *x1 = *x1_ct{{[0-9]+}};
// cublasDrotmg-NEXT:     dpct::get_default_queue().memcpy(param, param_ct{{[0-9]+}}, sizeof(double)*5).wait();
// cublasDrotmg-NEXT:     sycl::free(d1_ct{{[0-9]+}}, dpct::get_default_queue());
// cublasDrotmg-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasStpmv | FileCheck %s -check-prefix=cublasStpmv
// cublasStpmv: CUDA API:
// cublasStpmv-NEXT:   cublasStpmv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasStpmv-NEXT:               trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasStpmv-NEXT:               n /*int*/, a /*const float **/, x /*float **/, incx /*int*/);
// cublasStpmv-NEXT: Is migrated to:
// cublasStpmv-NEXT:   oneapi::mkl::blas::column_major::tpmv(handle->get_queue(), upper_lower, trans, unit_nonunit, n, a, x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasStrsv | FileCheck %s -check-prefix=cublasStrsv
// cublasStrsv: CUDA API:
// cublasStrsv-NEXT:   cublasStrsv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasStrsv-NEXT:               trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasStrsv-NEXT:               n /*int*/, a /*const float **/, lda /*int*/, x /*float **/,
// cublasStrsv-NEXT:               incx /*int*/);
// cublasStrsv-NEXT: Is migrated to:
// cublasStrsv-NEXT:   oneapi::mkl::blas::column_major::trsv(handle->get_queue(), upper_lower, trans, unit_nonunit, n, a, lda, x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasStbmv | FileCheck %s -check-prefix=cublasStbmv
// cublasStbmv: CUDA API:
// cublasStbmv-NEXT:   cublasStbmv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasStbmv-NEXT:               trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasStbmv-NEXT:               n /*int*/, k /*int*/, a /*const float **/, lda /*int*/,
// cublasStbmv-NEXT:               x /*float **/, incx /*int*/);
// cublasStbmv-NEXT: Is migrated to:
// cublasStbmv-NEXT:   oneapi::mkl::blas::column_major::tbmv(handle->get_queue(), upper_lower, trans, unit_nonunit, n, k, a, lda, x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasStpsv | FileCheck %s -check-prefix=cublasStpsv
// cublasStpsv: CUDA API:
// cublasStpsv-NEXT:   cublasStpsv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasStpsv-NEXT:               trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasStpsv-NEXT:               n /*int*/, a /*const float **/, x /*float **/, incx /*int*/);
// cublasStpsv-NEXT: Is migrated to:
// cublasStpsv-NEXT:   oneapi::mkl::blas::column_major::tpsv(handle->get_queue(), upper_lower, trans, unit_nonunit, n, a, x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCherkx | FileCheck %s -check-prefix=cublasCherkx
// cublasCherkx: CUDA API:
// cublasCherkx-NEXT:   cublasCherkx(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCherkx-NEXT:                trans /*cublasOperation_t*/, n /*int*/, k /*int*/,
// cublasCherkx-NEXT:                alpha /*const cuComplex **/, a /*const cuComplex **/,
// cublasCherkx-NEXT:                lda /*int*/, b /*const cuComplex **/, ldb /*int*/,
// cublasCherkx-NEXT:                beta /*const float **/, c /*cuComplex **/, ldc /*int*/);
// cublasCherkx-NEXT: Is migrated to:
// cublasCherkx-NEXT:   dpct::herk(handle->get_queue(), upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCgeam | FileCheck %s -check-prefix=cublasCgeam
// cublasCgeam: CUDA API:
// cublasCgeam-NEXT:   cublasCgeam(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasCgeam-NEXT:               transb /*cublasOperation_t*/, m /*int*/, n /*int*/,
// cublasCgeam-NEXT:               alpha /*const cuComplex **/, a /*const cuComplex **/, lda /*int*/,
// cublasCgeam-NEXT:               beta /*const cuComplex **/, b /*const cuComplex **/, ldb /*int*/,
// cublasCgeam-NEXT:               c /*cuComplex **/, ldc /*int*/);
// cublasCgeam-NEXT: Is migrated to:
// cublasCgeam-NEXT:   oneapi::mkl::blas::column_major::omatadd(handle->get_queue(), transa, transb, m, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)a, lda, dpct::get_value(beta, handle->get_queue()), (std::complex<float>*)b, ldb, (std::complex<float>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZrotg | FileCheck %s -check-prefix=cublasZrotg
// cublasZrotg: CUDA API:
// cublasZrotg-NEXT:   cublasZrotg(handle /*cublasHandle_t*/, a /*cuDoubleComplex **/,
// cublasZrotg-NEXT:               b /*cuDoubleComplex **/, c /*double **/, s /*cuDoubleComplex **/);
// cublasZrotg-NEXT: Is migrated to:
// cublasZrotg-NEXT:   sycl::double2* a_ct{{[0-9]+}} = a;
// cublasZrotg-NEXT:   sycl::double2* b_ct{{[0-9]+}} = b;
// cublasZrotg-NEXT:   double* c_ct{{[0-9]+}} = c;
// cublasZrotg-NEXT:   sycl::double2* s_ct{{[0-9]+}} = s;
// cublasZrotg-NEXT:   if(sycl::get_pointer_type(a, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(a, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
// cublasZrotg-NEXT:     a_ct{{[0-9]+}} = sycl::malloc_shared<sycl::double2>(3, dpct::get_default_queue());
// cublasZrotg-NEXT:     c_ct{{[0-9]+}} = sycl::malloc_shared<double>(1, dpct::get_default_queue());
// cublasZrotg-NEXT:     b_ct{{[0-9]+}} = a_ct{{[0-9]+}} + 1;
// cublasZrotg-NEXT:     s_ct{{[0-9]+}} = a_ct{{[0-9]+}} + 2;
// cublasZrotg-NEXT:     *a_ct{{[0-9]+}} = *a;
// cublasZrotg-NEXT:     *b_ct{{[0-9]+}} = *b;
// cublasZrotg-NEXT:     *c_ct{{[0-9]+}} = *c;
// cublasZrotg-NEXT:     *s_ct{{[0-9]+}} = *s;
// cublasZrotg-NEXT:   }
// cublasZrotg-NEXT:   oneapi::mkl::blas::column_major::rotg(handle->get_queue(), (std::complex<double>*)a_ct{{[0-9]+}}, (std::complex<double>*)b_ct{{[0-9]+}}, c_ct{{[0-9]+}}, (std::complex<double>*)s_ct{{[0-9]+}});
// cublasZrotg-NEXT:   if(sycl::get_pointer_type(a, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(a, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
// cublasZrotg-NEXT:     handle->get_queue().wait();
// cublasZrotg-NEXT:     *a = *a_ct{{[0-9]+}};
// cublasZrotg-NEXT:     *b = *b_ct{{[0-9]+}};
// cublasZrotg-NEXT:     *c = *c_ct{{[0-9]+}};
// cublasZrotg-NEXT:     *s = *s_ct{{[0-9]+}};
// cublasZrotg-NEXT:     sycl::free(a_ct{{[0-9]+}}, dpct::get_default_queue());
// cublasZrotg-NEXT:     sycl::free(c_ct{{[0-9]+}}, dpct::get_default_queue());
// cublasZrotg-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasHgemm | FileCheck %s -check-prefix=cublasHgemm
// cublasHgemm: CUDA API:
// cublasHgemm-NEXT:   cublasHgemm(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasHgemm-NEXT:               transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
// cublasHgemm-NEXT:               alpha /*const __half **/, a /*const __half **/, lda /*int*/,
// cublasHgemm-NEXT:               b /*const __half **/, ldb /*int*/, beta /*const __half **/,
// cublasHgemm-NEXT:               c /*__half **/, ldc /*int*/);
// cublasHgemm-NEXT: Is migrated to:
// cublasHgemm-NEXT:   oneapi::mkl::blas::column_major::gemm(handle->get_queue(), transa, transb, m, n, k, dpct::get_value(alpha, handle->get_queue()), a, lda, b, ldb, dpct::get_value(beta, handle->get_queue()), c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSgemv | FileCheck %s -check-prefix=cublasSgemv
// cublasSgemv: CUDA API:
// cublasSgemv-NEXT:   cublasSgemv(handle /*cublasHandle_t*/, trans /*cublasOperation_t*/, m /*int*/,
// cublasSgemv-NEXT:               n /*int*/, alpha /*const float **/, a /*const float **/,
// cublasSgemv-NEXT:               lda /*int*/, x /*const float **/, incx /*int*/,
// cublasSgemv-NEXT:               beta /*const float **/, y /*float **/, incy /*int*/);
// cublasSgemv-NEXT: Is migrated to:
// cublasSgemv-NEXT:   oneapi::mkl::blas::column_major::gemv(handle->get_queue(), trans, m, n, dpct::get_value(alpha, handle->get_queue()), a, lda, x, incx, dpct::get_value(beta, handle->get_queue()), y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSrotmg | FileCheck %s -check-prefix=cublasSrotmg
// cublasSrotmg: CUDA API:
// cublasSrotmg-NEXT:   cublasSrotmg(handle /*cublasHandle_t*/, d1 /*float **/, d2 /*float **/,
// cublasSrotmg-NEXT:                x1 /*float **/, y1 /*const float **/, param /*float **/);
// cublasSrotmg-NEXT: Is migrated to:
// cublasSrotmg-NEXT:   float* d1_ct{{[0-9]+}} = d1;
// cublasSrotmg-NEXT:   float* d2_ct{{[0-9]+}} = d2;
// cublasSrotmg-NEXT:   float* x1_ct{{[0-9]+}} = x1;
// cublasSrotmg-NEXT:   float* param_ct{{[0-9]+}} = param;
// cublasSrotmg-NEXT:   if(sycl::get_pointer_type(d1, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(d1, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
// cublasSrotmg-NEXT:     d1_ct{{[0-9]+}} = sycl::malloc_shared<float>(8, dpct::get_default_queue());
// cublasSrotmg-NEXT:     d2_ct{{[0-9]+}} = d1_ct{{[0-9]+}} + 1;
// cublasSrotmg-NEXT:     x1_ct{{[0-9]+}} = d1_ct{{[0-9]+}} + 2;
// cublasSrotmg-NEXT:     param_ct{{[0-9]+}} = d1_ct{{[0-9]+}} + 3;
// cublasSrotmg-NEXT:     *d1_ct{{[0-9]+}} = *d1;
// cublasSrotmg-NEXT:     *d2_ct{{[0-9]+}} = *d2;
// cublasSrotmg-NEXT:     *x1_ct{{[0-9]+}} = *x1;
// cublasSrotmg-NEXT:   }
// cublasSrotmg-NEXT:   oneapi::mkl::blas::column_major::rotmg(handle->get_queue(), d1_ct{{[0-9]+}}, d2_ct{{[0-9]+}}, x1_ct{{[0-9]+}}, dpct::get_value(y1, handle->get_queue()), param_ct{{[0-9]+}});
// cublasSrotmg-NEXT:   if(sycl::get_pointer_type(d1, handle->get_queue().get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(d1, handle->get_queue().get_context())!=sycl::usm::alloc::shared) {
// cublasSrotmg-NEXT:     handle->get_queue().wait();
// cublasSrotmg-NEXT:     *d1 = *d1_ct{{[0-9]+}};
// cublasSrotmg-NEXT:     *d2 = *d2_ct{{[0-9]+}};
// cublasSrotmg-NEXT:     *x1 = *x1_ct{{[0-9]+}};
// cublasSrotmg-NEXT:     dpct::get_default_queue().memcpy(param, param_ct{{[0-9]+}}, sizeof(float)*5).wait();
// cublasSrotmg-NEXT:     sycl::free(d1_ct{{[0-9]+}}, dpct::get_default_queue());
// cublasSrotmg-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZhbmv | FileCheck %s -check-prefix=cublasZhbmv
// cublasZhbmv: CUDA API:
// cublasZhbmv-NEXT:   cublasZhbmv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZhbmv-NEXT:               n /*int*/, k /*int*/, alpha /*const cuDoubleComplex **/,
// cublasZhbmv-NEXT:               a /*const cuDoubleComplex **/, lda /*int*/,
// cublasZhbmv-NEXT:               x /*const cuDoubleComplex **/, incx /*int*/,
// cublasZhbmv-NEXT:               beta /*const cuDoubleComplex **/, y /*cuDoubleComplex **/,
// cublasZhbmv-NEXT:               incy /*int*/);
// cublasZhbmv-NEXT: Is migrated to:
// cublasZhbmv-NEXT:   oneapi::mkl::blas::column_major::hbmv(handle->get_queue(), upper_lower, n, k, dpct::get_value(alpha, handle->get_queue()), (std::complex<double>*)a, lda, (std::complex<double>*)x, incx, dpct::get_value(beta, handle->get_queue()), (std::complex<double>*)y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCswap | FileCheck %s -check-prefix=cublasCswap
// cublasCswap: CUDA API:
// cublasCswap-NEXT:   cublasCswap(handle /*cublasHandle_t*/, n /*int*/, x /*cuComplex **/,
// cublasCswap-NEXT:               incx /*int*/, y /*cuComplex **/, incy /*int*/);
// cublasCswap-NEXT: Is migrated to:
// cublasCswap-NEXT:   oneapi::mkl::blas::column_major::swap(handle->get_queue(), n, (std::complex<float>*)x, incx, (std::complex<float>*)y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCher2 | FileCheck %s -check-prefix=cublasCher2
// cublasCher2: CUDA API:
// cublasCher2-NEXT:   cublasCher2(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCher2-NEXT:               n /*int*/, alpha /*const cuComplex **/, x /*const cuComplex **/,
// cublasCher2-NEXT:               incx /*int*/, y /*const cuComplex **/, incy /*int*/,
// cublasCher2-NEXT:               a /*cuComplex **/, lda /*int*/);
// cublasCher2-NEXT: Is migrated to:
// cublasCher2-NEXT:   oneapi::mkl::blas::column_major::her2(handle->get_queue(), upper_lower, n, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)x, incx, (std::complex<float>*)y, incy, (std::complex<float>*)a, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCaxpy | FileCheck %s -check-prefix=cublasCaxpy
// cublasCaxpy: CUDA API:
// cublasCaxpy-NEXT:   cublasCaxpy(handle /*cublasHandle_t*/, n /*int*/, alpha /*const cuComplex **/,
// cublasCaxpy-NEXT:               x /*const cuComplex **/, incx /*int*/, y /*cuComplex **/,
// cublasCaxpy-NEXT:               incy /*int*/);
// cublasCaxpy-NEXT: Is migrated to:
// cublasCaxpy-NEXT:   oneapi::mkl::blas::column_major::axpy(handle->get_queue(), n, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)x, incx, (std::complex<float>*)y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDaxpy | FileCheck %s -check-prefix=cublasDaxpy
// cublasDaxpy: CUDA API:
// cublasDaxpy-NEXT:   cublasDaxpy(handle /*cublasHandle_t*/, n /*int*/, alpha /*const double **/,
// cublasDaxpy-NEXT:               x /*const double **/, incx /*int*/, y /*double **/, incy /*int*/);
// cublasDaxpy-NEXT: Is migrated to:
// cublasDaxpy-NEXT:   oneapi::mkl::blas::column_major::axpy(handle->get_queue(), n, dpct::get_value(alpha, handle->get_queue()), x, incx, y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDscal | FileCheck %s -check-prefix=cublasDscal
// cublasDscal: CUDA API:
// cublasDscal-NEXT:   cublasDscal(handle /*cublasHandle_t*/, n /*int*/, alpha /*const double **/,
// cublasDscal-NEXT:               x /*double **/, incx /*int*/);
// cublasDscal-NEXT: Is migrated to:
// cublasDscal-NEXT:   oneapi::mkl::blas::column_major::scal(handle->get_queue(), n, dpct::get_value(alpha, handle->get_queue()), x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCtrmm | FileCheck %s -check-prefix=cublasCtrmm
// cublasCtrmm: CUDA API:
// cublasCtrmm-NEXT:   cublasCtrmm(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasCtrmm-NEXT:               upper_lower /*cublasFillMode_t*/, transa /*cublasOperation_t*/,
// cublasCtrmm-NEXT:               unit_diag /*cublasDiagType_t*/, m /*int*/, n /*int*/,
// cublasCtrmm-NEXT:               alpha /*const cuComplex **/, a /*const cuComplex **/, lda /*int*/,
// cublasCtrmm-NEXT:               b /*const cuComplex **/, ldb /*int*/, c /*cuComplex **/,
// cublasCtrmm-NEXT:               ldc /*int*/);
// cublasCtrmm-NEXT: Is migrated to:
// cublasCtrmm-NEXT:   dpct::trmm(handle->get_queue(), left_right, upper_lower, transa, unit_diag, m, n, alpha, a, lda, b, ldb, c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCgbmv | FileCheck %s -check-prefix=cublasCgbmv
// cublasCgbmv: CUDA API:
// cublasCgbmv-NEXT:   cublasCgbmv(handle /*cublasHandle_t*/, trans /*cublasOperation_t*/, m /*int*/,
// cublasCgbmv-NEXT:               n /*int*/, kl /*int*/, ku /*int*/, alpha /*const cuComplex **/,
// cublasCgbmv-NEXT:               a /*const cuComplex **/, lda /*int*/, x /*const cuComplex **/,
// cublasCgbmv-NEXT:               incx /*int*/, beta /*const cuComplex **/, y /*cuComplex **/,
// cublasCgbmv-NEXT:               incy /*int*/);
// cublasCgbmv-NEXT: Is migrated to:
// cublasCgbmv-NEXT:   oneapi::mkl::blas::column_major::gbmv(handle->get_queue(), trans, m, n, kl, ku, dpct::get_value(alpha, handle->get_queue()), (std::complex<float>*)a, lda, (std::complex<float>*)x, incx, dpct::get_value(beta, handle->get_queue()), (std::complex<float>*)y, incy);
