// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZdscal | FileCheck %s -check-prefix=cublasZdscal
// cublasZdscal: CUDA API:
// cublasZdscal-NEXT:   cublasZdscal(handle /*cublasHandle_t*/, n /*int*/, alpha /*const double **/,
// cublasZdscal-NEXT:                x /*cuDoubleComplex **/, incx /*int*/);
// cublasZdscal-NEXT: Is migrated to:
// cublasZdscal-NEXT:   oneapi::mkl::blas::column_major::scal(*handle, n, dpct::get_value(alpha, *handle), (std::complex<double>*)x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasGetVectorAsync | FileCheck %s -check-prefix=cublasGetVectorAsync
// cublasGetVectorAsync: CUDA API:
// cublasGetVectorAsync-NEXT:   cublasGetVectorAsync(n /*int*/, elementsize /*int*/, from /*const void **/,
// cublasGetVectorAsync-NEXT:                        incx /*int*/, to /*void **/, incy /*int*/,
// cublasGetVectorAsync-NEXT:                        stream /*cudaStream_t*/);
// cublasGetVectorAsync-NEXT: Is migrated to:
// cublasGetVectorAsync-NEXT:   dpct::matrix_mem_copy((void*)to, (void*)from, incy, incx, 1, n, elementsize, dpct::automatic, *stream, true);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZsyrk | FileCheck %s -check-prefix=cublasZsyrk
// cublasZsyrk: CUDA API:
// cublasZsyrk-NEXT:   cublasZsyrk(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZsyrk-NEXT:               trans /*cublasOperation_t*/, n /*int*/, k /*int*/,
// cublasZsyrk-NEXT:               alpha /*const cuDoubleComplex **/, a /*const cuDoubleComplex **/,
// cublasZsyrk-NEXT:               lda /*int*/, beta /*const cuDoubleComplex **/,
// cublasZsyrk-NEXT:               c /*cuDoubleComplex **/, ldc /*int*/);
// cublasZsyrk-NEXT: Is migrated to:
// cublasZsyrk-NEXT:   oneapi::mkl::blas::column_major::syrk(*handle, upper_lower, trans, n, k, dpct::get_value(alpha, *handle), (std::complex<double>*)a, lda, dpct::get_value(beta, *handle), (std::complex<double>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDtrsm | FileCheck %s -check-prefix=cublasDtrsm
// cublasDtrsm: CUDA API:
// cublasDtrsm-NEXT:   cublasDtrsm(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasDtrsm-NEXT:               upper_lower /*cublasFillMode_t*/, transa /*cublasOperation_t*/,
// cublasDtrsm-NEXT:               unit_diag /*cublasDiagType_t*/, m /*int*/, n /*int*/,
// cublasDtrsm-NEXT:               alpha /*const double **/, a /*const double **/, lda /*int*/,
// cublasDtrsm-NEXT:               b /*double **/, ldb /*int*/);
// cublasDtrsm-NEXT: Is migrated to:
// cublasDtrsm-NEXT:   oneapi::mkl::blas::column_major::trsm(*handle, left_right, upper_lower, transa, unit_diag, m, n, dpct::get_value(alpha, *handle), a, lda, b, ldb);

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
// cublasCgemmStridedBatched-NEXT:   oneapi::mkl::blas::column_major::gemm_batch(*handle, transa, transb, m, n, k, dpct::get_value(alpha, *handle), (std::complex<float>*)a, lda, stridea, (std::complex<float>*)b, ldb, strideb, dpct::get_value(beta, *handle), (std::complex<float>*)c, ldc, stridec, group_count);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasChemm | FileCheck %s -check-prefix=cublasChemm
// cublasChemm: CUDA API:
// cublasChemm-NEXT:   cublasChemm(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasChemm-NEXT:               upper_lower /*cublasFillMode_t*/, m /*int*/, n /*int*/,
// cublasChemm-NEXT:               alpha /*const cuComplex **/, a /*const cuComplex **/, lda /*int*/,
// cublasChemm-NEXT:               b /*const cuComplex **/, ldb /*int*/, beta /*const cuComplex **/,
// cublasChemm-NEXT:               c /*cuComplex **/, ldc /*int*/);
// cublasChemm-NEXT: Is migrated to:
// cublasChemm-NEXT:   oneapi::mkl::blas::column_major::hemm(*handle, left_right, upper_lower, m, n, dpct::get_value(alpha, *handle), (std::complex<float>*)a, lda, (std::complex<float>*)b, ldb, dpct::get_value(beta, *handle), (std::complex<float>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDrotg | FileCheck %s -check-prefix=cublasDrotg
// cublasDrotg: CUDA API:
// cublasDrotg-NEXT:   cublasDrotg(handle /*cublasHandle_t*/, a /*double **/, b /*double **/,
// cublasDrotg-NEXT:               c /*double **/, s /*double **/);
// cublasDrotg-NEXT: Is migrated to:
// cublasDrotg-NEXT:   double* a_ct{{[0-9]+}} = a;
// cublasDrotg-NEXT:   double* b_ct{{[0-9]+}} = b;
// cublasDrotg-NEXT:   double* c_ct{{[0-9]+}} = c;
// cublasDrotg-NEXT:   double* s_ct{{[0-9]+}} = s;
// cublasDrotg-NEXT:   if(sycl::get_pointer_type(a, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(a, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasDrotg-NEXT:     a_ct{{[0-9]+}} = sycl::malloc_shared<double>(4, dpct::get_default_queue());
// cublasDrotg-NEXT:     b_ct{{[0-9]+}} = a_ct{{[0-9]+}} + 1;
// cublasDrotg-NEXT:     c_ct{{[0-9]+}} = a_ct{{[0-9]+}} + 2;
// cublasDrotg-NEXT:     s_ct{{[0-9]+}} = a_ct{{[0-9]+}} + 3;
// cublasDrotg-NEXT:     *a_ct{{[0-9]+}} = *a;
// cublasDrotg-NEXT:     *b_ct{{[0-9]+}} = *b;
// cublasDrotg-NEXT:     *c_ct{{[0-9]+}} = *c;
// cublasDrotg-NEXT:     *s_ct{{[0-9]+}} = *s;
// cublasDrotg-NEXT:   }
// cublasDrotg-NEXT:   oneapi::mkl::blas::column_major::rotg(*handle, a_ct{{[0-9]+}}, b_ct{{[0-9]+}}, c_ct{{[0-9]+}}, s_ct{{[0-9]+}});
// cublasDrotg-NEXT:   if(sycl::get_pointer_type(a, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(a, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasDrotg-NEXT:     handle->wait();
// cublasDrotg-NEXT:     *a = *a_ct{{[0-9]+}};
// cublasDrotg-NEXT:     *b = *b_ct{{[0-9]+}};
// cublasDrotg-NEXT:     *c = *c_ct{{[0-9]+}};
// cublasDrotg-NEXT:     *s = *s_ct{{[0-9]+}};
// cublasDrotg-NEXT:     sycl::free(a_ct{{[0-9]+}}, dpct::get_default_queue());
// cublasDrotg-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCtpsv | FileCheck %s -check-prefix=cublasCtpsv
// cublasCtpsv: CUDA API:
// cublasCtpsv-NEXT:   cublasCtpsv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCtpsv-NEXT:               trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasCtpsv-NEXT:               n /*int*/, a /*const cuComplex **/, x /*cuComplex **/,
// cublasCtpsv-NEXT:               incx /*int*/);
// cublasCtpsv-NEXT: Is migrated to:
// cublasCtpsv-NEXT:   oneapi::mkl::blas::column_major::tpsv(*handle, upper_lower, trans, unit_nonunit, n, (std::complex<float>*)a, (std::complex<float>*)x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasAxpyEx | FileCheck %s -check-prefix=cublasAxpyEx
// cublasAxpyEx: CUDA API:
// cublasAxpyEx-NEXT:   cublasAxpyEx(handle /*cublasHandle_t*/, n /*int*/, alpha /*const void **/,
// cublasAxpyEx-NEXT:                alphatype /*cudaDataType*/, x /*const void **/,
// cublasAxpyEx-NEXT:                xtype /*cudaDataType*/, incx /*int*/, y /*void **/,
// cublasAxpyEx-NEXT:                ytype /*cudaDataType*/, incy /*int*/,
// cublasAxpyEx-NEXT:                computetype /*cudaDataType*/);
// cublasAxpyEx-NEXT: Is migrated to:
// cublasAxpyEx-NEXT:   dpct::axpy(*handle, n, alpha, alphatype, x, xtype, incx, y, ytype, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDtpmv | FileCheck %s -check-prefix=cublasDtpmv
// cublasDtpmv: CUDA API:
// cublasDtpmv-NEXT:   cublasDtpmv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDtpmv-NEXT:               trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasDtpmv-NEXT:               n /*int*/, a /*const double **/, x /*double **/, incx /*int*/);
// cublasDtpmv-NEXT: Is migrated to:
// cublasDtpmv-NEXT:   oneapi::mkl::blas::column_major::tpmv(*handle, upper_lower, trans, unit_nonunit, n, a, x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZtrmv | FileCheck %s -check-prefix=cublasZtrmv
// cublasZtrmv: CUDA API:
// cublasZtrmv-NEXT:   cublasZtrmv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZtrmv-NEXT:               trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasZtrmv-NEXT:               n /*int*/, a /*const cuDoubleComplex **/, lda /*int*/,
// cublasZtrmv-NEXT:               x /*cuDoubleComplex **/, incx /*int*/);
// cublasZtrmv-NEXT: Is migrated to:
// cublasZtrmv-NEXT:   oneapi::mkl::blas::column_major::trmv(*handle, upper_lower, trans, unit_nonunit, n, (std::complex<double>*)a, lda, (std::complex<double>*)x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSdot | FileCheck %s -check-prefix=cublasSdot
// cublasSdot: CUDA API:
// cublasSdot-NEXT:   cublasSdot(handle /*cublasHandle_t*/, n /*int*/, x /*const float **/,
// cublasSdot-NEXT:              incx /*int*/, y /*const float **/, incy /*int*/, res /*float **/);
// cublasSdot-NEXT: Is migrated to:
// cublasSdot-NEXT:   float* res_temp_ptr_ct{{[0-9]+}} = res;
// cublasSdot-NEXT:   if(sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasSdot-NEXT:     res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<float>(1, dpct::get_default_queue());
// cublasSdot-NEXT:   }
// cublasSdot-NEXT:   oneapi::mkl::blas::column_major::dot(*handle, n, x, incx, y, incy, res_temp_ptr_ct{{[0-9]+}});
// cublasSdot-NEXT:   if(sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasSdot-NEXT:     handle->wait();
// cublasSdot-NEXT:     *res = *res_temp_ptr_ct{{[0-9]+}};
// cublasSdot-NEXT:     sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_default_queue());
// cublasSdot-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasGetVector | FileCheck %s -check-prefix=cublasGetVector
// cublasGetVector: CUDA API:
// cublasGetVector-NEXT:   cublasGetVector(n /*int*/, elementsize /*int*/, x /*const void **/,
// cublasGetVector-NEXT:                   incx /*int*/, y /*void **/, incy /*int*/);
// cublasGetVector-NEXT: Is migrated to:
// cublasGetVector-NEXT:   dpct::matrix_mem_copy((void*)y, (void*)x, incy, incx, 1, n, elementsize);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDgemv | FileCheck %s -check-prefix=cublasDgemv
// cublasDgemv: CUDA API:
// cublasDgemv-NEXT:   cublasDgemv(handle /*cublasHandle_t*/, trans /*cublasOperation_t*/, m /*int*/,
// cublasDgemv-NEXT:               n /*int*/, alpha /*const double **/, a /*const double **/,
// cublasDgemv-NEXT:               lda /*int*/, x /*const double **/, incx /*int*/,
// cublasDgemv-NEXT:               beta /*const double **/, y /*double **/, incy /*int*/);
// cublasDgemv-NEXT: Is migrated to:
// cublasDgemv-NEXT:   oneapi::mkl::blas::column_major::gemv(*handle, trans, m, n, dpct::get_value(alpha, *handle), a, lda, x, incx, dpct::get_value(beta, *handle), y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasChpr | FileCheck %s -check-prefix=cublasChpr
// cublasChpr: CUDA API:
// cublasChpr-NEXT:   cublasChpr(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasChpr-NEXT:              n /*int*/, alpha /*const float **/, x /*const cuComplex **/,
// cublasChpr-NEXT:              incx /*int*/, a /*cuComplex **/);
// cublasChpr-NEXT: Is migrated to:
// cublasChpr-NEXT:   oneapi::mkl::blas::column_major::hpr(*handle, upper_lower, n, dpct::get_value(alpha, *handle), (std::complex<float>*)x, incx, (std::complex<float>*)a);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZgeru | FileCheck %s -check-prefix=cublasZgeru
// cublasZgeru: CUDA API:
// cublasZgeru-NEXT:   cublasZgeru(handle /*cublasHandle_t*/, m /*int*/, n /*int*/,
// cublasZgeru-NEXT:               alpha /*const cuDoubleComplex **/, x /*const cuDoubleComplex **/,
// cublasZgeru-NEXT:               incx /*int*/, y /*const cuDoubleComplex **/, incy /*int*/,
// cublasZgeru-NEXT:               a /*cuDoubleComplex **/, lda /*int*/);
// cublasZgeru-NEXT: Is migrated to:
// cublasZgeru-NEXT:   oneapi::mkl::blas::column_major::geru(*handle, m, n, dpct::get_value(alpha, *handle), (std::complex<double>*)x, incx, (std::complex<double>*)y, incy, (std::complex<double>*)a, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCdgmm | FileCheck %s -check-prefix=cublasCdgmm
// cublasCdgmm: CUDA API:
// cublasCdgmm-NEXT:   cublasCdgmm(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasCdgmm-NEXT:               m /*int*/, n /*int*/, a /*const cuComplex **/, lda /*int*/,
// cublasCdgmm-NEXT:               x /*const cuComplex **/, incx /*int*/, c /*cuComplex **/,
// cublasCdgmm-NEXT:               ldc /*int*/);
// cublasCdgmm-NEXT: Is migrated to:
// cublasCdgmm-NEXT:   oneapi::mkl::blas::column_major::dgmm_batch(*handle, left_right, m, n, (std::complex<float>*)a, lda, 0, (std::complex<float>*)x, incx, 0, (std::complex<float>*)c, ldc, ldc * n, 1);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDcopy | FileCheck %s -check-prefix=cublasDcopy
// cublasDcopy: CUDA API:
// cublasDcopy-NEXT:   cublasDcopy(handle /*cublasHandle_t*/, n /*int*/, x /*const double **/,
// cublasDcopy-NEXT:               incx /*int*/, y /*double **/, incy /*int*/);
// cublasDcopy-NEXT: Is migrated to:
// cublasDcopy-NEXT:   oneapi::mkl::blas::column_major::copy(*handle, n, x, incx, y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSsyrkx | FileCheck %s -check-prefix=cublasSsyrkx
// cublasSsyrkx: CUDA API:
// cublasSsyrkx-NEXT:   cublasSsyrkx(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasSsyrkx-NEXT:                trans /*cublasOperation_t*/, n /*int*/, k /*int*/,
// cublasSsyrkx-NEXT:                alpha /*const float **/, a /*const float **/, lda /*int*/,
// cublasSsyrkx-NEXT:                b /*const float **/, ldb /*int*/, beta /*const float **/,
// cublasSsyrkx-NEXT:                c /*float **/, ldc /*int*/);
// cublasSsyrkx-NEXT: Is migrated to:
// cublasSsyrkx-NEXT:   dpct::syrk(*handle, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDestroy | FileCheck %s -check-prefix=cublasDestroy
// cublasDestroy: CUDA API:
// cublasDestroy-NEXT:   cublasDestroy(handle /*cublasHandle_t*/);
// cublasDestroy-NEXT: Is migrated to:
// cublasDestroy-NEXT:   handle = nullptr;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasGetAtomicsMode | FileCheck %s -check-prefix=cublasGetAtomicsMode
// cublasGetAtomicsMode: CUDA API:
// cublasGetAtomicsMode-NEXT:   cublasGetAtomicsMode(handle /*cublasHandle_t*/,
// cublasGetAtomicsMode-NEXT:                        atomics /*cublasAtomicsMode_t **/);
// cublasGetAtomicsMode-NEXT: Is migrated to:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZsyr2k | FileCheck %s -check-prefix=cublasZsyr2k
// cublasZsyr2k: CUDA API:
// cublasZsyr2k-NEXT:   cublasZsyr2k(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZsyr2k-NEXT:                trans /*cublasOperation_t*/, n /*int*/, k /*int*/,
// cublasZsyr2k-NEXT:                alpha /*const cuDoubleComplex **/, a /*const cuDoubleComplex **/,
// cublasZsyr2k-NEXT:                lda /*int*/, b /*const cuDoubleComplex **/, ldb /*int*/,
// cublasZsyr2k-NEXT:                beta /*const cuDoubleComplex **/, c /*cuDoubleComplex **/,
// cublasZsyr2k-NEXT:                ldc /*int*/);
// cublasZsyr2k-NEXT: Is migrated to:
// cublasZsyr2k-NEXT:   oneapi::mkl::blas::column_major::syr2k(*handle, upper_lower, trans, n, k, dpct::get_value(alpha, *handle), (std::complex<double>*)a, lda, (std::complex<double>*)b, ldb, dpct::get_value(beta, *handle), (std::complex<double>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDrotm | FileCheck %s -check-prefix=cublasDrotm
// cublasDrotm: CUDA API:
// cublasDrotm-NEXT:   cublasDrotm(handle /*cublasHandle_t*/, n /*int*/, x /*double **/,
// cublasDrotm-NEXT:               incx /*int*/, y /*double **/, incy /*int*/,
// cublasDrotm-NEXT:               param /*const double **/);
// cublasDrotm-NEXT: Is migrated to:
// cublasDrotm-NEXT:   oneapi::mkl::blas::column_major::rotm(*handle, n, x, incx, y, incy, const_cast<double*>(param));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCdotu | FileCheck %s -check-prefix=cublasCdotu
// cublasCdotu: CUDA API:
// cublasCdotu-NEXT:   cublasCdotu(handle /*cublasHandle_t*/, n /*int*/, x /*const cuComplex **/,
// cublasCdotu-NEXT:               incx /*int*/, y /*const cuComplex **/, incy /*int*/,
// cublasCdotu-NEXT:               res /*cuComplex **/);
// cublasCdotu-NEXT: Is migrated to:
// cublasCdotu-NEXT:   sycl::float2* res_temp_ptr_ct{{[0-9]+}} = res;
// cublasCdotu-NEXT:   if(sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasCdotu-NEXT:     res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<sycl::float2>(1, dpct::get_default_queue());
// cublasCdotu-NEXT:   }
// cublasCdotu-NEXT:   oneapi::mkl::blas::column_major::dotu(*handle, n, (std::complex<float>*)x, incx, (std::complex<float>*)y, incy, (std::complex<float>*)res_temp_ptr_ct{{[0-9]+}});
// cublasCdotu-NEXT:   if(sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasCdotu-NEXT:     handle->wait();
// cublasCdotu-NEXT:     *res = *res_temp_ptr_ct{{[0-9]+}};
// cublasCdotu-NEXT:     sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_default_queue());
// cublasCdotu-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDspr | FileCheck %s -check-prefix=cublasDspr
// cublasDspr: CUDA API:
// cublasDspr-NEXT:   cublasDspr(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDspr-NEXT:              n /*int*/, alpha /*const double **/, x /*const double **/,
// cublasDspr-NEXT:              incx /*int*/, a /*double **/);
// cublasDspr-NEXT: Is migrated to:
// cublasDspr-NEXT:   oneapi::mkl::blas::column_major::spr(*handle, upper_lower, n, dpct::get_value(alpha, *handle), x, incx, a);

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
// cublasSasum-NEXT:     res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<float>(1, dpct::get_default_queue());
// cublasSasum-NEXT:   }
// cublasSasum-NEXT:   oneapi::mkl::blas::column_major::asum(*handle, n, x, incx, res_temp_ptr_ct{{[0-9]+}});
// cublasSasum-NEXT:   if(sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasSasum-NEXT:     handle->wait();
// cublasSasum-NEXT:     *res = *res_temp_ptr_ct{{[0-9]+}};
// cublasSasum-NEXT:     sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_default_queue());
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
// cublasCrotg-NEXT:     a_ct{{[0-9]+}} = sycl::malloc_shared<sycl::float2>(3, dpct::get_default_queue());
// cublasCrotg-NEXT:     c_ct{{[0-9]+}} = sycl::malloc_shared<float>(1, dpct::get_default_queue());
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
// cublasCrotg-NEXT:     sycl::free(a_ct{{[0-9]+}}, dpct::get_default_queue());
// cublasCrotg-NEXT:     sycl::free(c_ct{{[0-9]+}}, dpct::get_default_queue());
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
// cublasIzamax-NEXT: Is migrated to:
// cublasIzamax-NEXT:   int64_t* res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<int64_t>(1, dpct::get_default_queue());
// cublasIzamax-NEXT:   oneapi::mkl::blas::column_major::iamax(*handle, n, (std::complex<double>*)x, incx, res_temp_ptr_ct{{[0-9]+}}, oneapi::mkl::index_base::one).wait();
// cublasIzamax-NEXT:   int res_temp_host_ct{{[0-9]+}} = (int)*res_temp_ptr_ct{{[0-9]+}};
// cublasIzamax-NEXT:   dpct::dpct_memcpy(res, &res_temp_host_ct{{[0-9]+}}, sizeof(int));
// cublasIzamax-NEXT:   sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_default_queue());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasScnrm2 | FileCheck %s -check-prefix=cublasScnrm2
// cublasScnrm2: CUDA API:
// cublasScnrm2-NEXT:   cublasScnrm2(handle /*cublasHandle_t*/, n /*int*/, x /*const cuComplex **/,
// cublasScnrm2-NEXT:                incx /*int*/, res /*float **/);
// cublasScnrm2-NEXT: Is migrated to:
// cublasScnrm2-NEXT:   float* res_temp_ptr_ct{{[0-9]+}} = res;
// cublasScnrm2-NEXT:   if(sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasScnrm2-NEXT:     res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<float>(1, dpct::get_default_queue());
// cublasScnrm2-NEXT:   }
// cublasScnrm2-NEXT:   oneapi::mkl::blas::column_major::nrm2(*handle, n, (std::complex<float>*)x, incx, res_temp_ptr_ct{{[0-9]+}});
// cublasScnrm2-NEXT:   if(sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasScnrm2-NEXT:     handle->wait();
// cublasScnrm2-NEXT:     *res = *res_temp_ptr_ct{{[0-9]+}};
// cublasScnrm2-NEXT:     sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_default_queue());
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

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDgeam | FileCheck %s -check-prefix=cublasDgeam
// cublasDgeam: CUDA API:
// cublasDgeam-NEXT:   cublasDgeam(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasDgeam-NEXT:               transb /*cublasOperation_t*/, m /*int*/, n /*int*/,
// cublasDgeam-NEXT:               alpha /*const double **/, a /*const double **/, lda /*int*/,
// cublasDgeam-NEXT:               beta /*const double **/, b /*const double **/, ldb /*int*/,
// cublasDgeam-NEXT:               c /*double **/, ldc /*int*/);
// cublasDgeam-NEXT: Is migrated to:
// cublasDgeam-NEXT:   oneapi::mkl::blas::column_major::omatadd(*handle, transa, transb, m, n, dpct::get_value(alpha, *handle), a, lda, dpct::get_value(beta, *handle), b, ldb, c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDotcEx | FileCheck %s -check-prefix=cublasDotcEx
// cublasDotcEx: CUDA API:
// cublasDotcEx-NEXT:   cublasDotcEx(handle /*cublasHandle_t*/, n /*int*/, x /*const void **/,
// cublasDotcEx-NEXT:                xtype /*cudaDataType*/, incx /*int*/, y /*const void **/,
// cublasDotcEx-NEXT:                ytype /*cudaDataType*/, incy /*int*/, res /*void **/,
// cublasDotcEx-NEXT:                restype /*cudaDataType*/, computetype /*cudaDataType*/);
// cublasDotcEx-NEXT: Is migrated to:
// cublasDotcEx-NEXT:   dpct::dotc(*handle, n, x, xtype, incx, y, ytype, incy, res, restype);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSsyr2 | FileCheck %s -check-prefix=cublasSsyr2
// cublasSsyr2: CUDA API:
// cublasSsyr2-NEXT:   cublasSsyr2(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasSsyr2-NEXT:               n /*int*/, alpha /*const float **/, x /*const float **/,
// cublasSsyr2-NEXT:               incx /*int*/, y /*const float **/, incy /*int*/, a /*float **/,
// cublasSsyr2-NEXT:               lda /*int*/);
// cublasSsyr2-NEXT: Is migrated to:
// cublasSsyr2-NEXT:   oneapi::mkl::blas::column_major::syr2(*handle, upper_lower, n, dpct::get_value(alpha, *handle), x, incx, y, incy, a, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasChemv | FileCheck %s -check-prefix=cublasChemv
// cublasChemv: CUDA API:
// cublasChemv-NEXT:   cublasChemv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasChemv-NEXT:               n /*int*/, alpha /*const cuComplex **/, a /*const cuComplex **/,
// cublasChemv-NEXT:               lda /*int*/, x /*const cuComplex **/, incx /*int*/,
// cublasChemv-NEXT:               beta /*const cuComplex **/, y /*cuComplex **/, incy /*int*/);
// cublasChemv-NEXT: Is migrated to:
// cublasChemv-NEXT:   oneapi::mkl::blas::column_major::hemv(*handle, upper_lower, n, dpct::get_value(alpha, *handle), (std::complex<float>*)a, lda, (std::complex<float>*)x, incx, dpct::get_value(beta, *handle), (std::complex<float>*)y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDsyrkx | FileCheck %s -check-prefix=cublasDsyrkx
// cublasDsyrkx: CUDA API:
// cublasDsyrkx-NEXT:   cublasDsyrkx(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDsyrkx-NEXT:                trans /*cublasOperation_t*/, n /*int*/, k /*int*/,
// cublasDsyrkx-NEXT:                alpha /*const double **/, a /*const double **/, lda /*int*/,
// cublasDsyrkx-NEXT:                b /*const double **/, ldb /*int*/, beta /*const double **/,
// cublasDsyrkx-NEXT:                c /*double **/, ldc /*int*/);
// cublasDsyrkx-NEXT: Is migrated to:
// cublasDsyrkx-NEXT:   dpct::syrk(*handle, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZtbmv | FileCheck %s -check-prefix=cublasZtbmv
// cublasZtbmv: CUDA API:
// cublasZtbmv-NEXT:   cublasZtbmv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZtbmv-NEXT:               trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasZtbmv-NEXT:               n /*int*/, k /*int*/, a /*const cuDoubleComplex **/, lda /*int*/,
// cublasZtbmv-NEXT:               x /*cuDoubleComplex **/, incx /*int*/);
// cublasZtbmv-NEXT: Is migrated to:
// cublasZtbmv-NEXT:   oneapi::mkl::blas::column_major::tbmv(*handle, upper_lower, trans, unit_nonunit, n, k, (std::complex<double>*)a, lda, (std::complex<double>*)x, incx);

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
// cublasHgemmStridedBatched-NEXT:   oneapi::mkl::blas::column_major::gemm_batch(*handle, transa, transb, m, n, k, dpct::get_value(alpha, *handle), a, lda, stridea, b, ldb, strideb, dpct::get_value(beta, *handle), c, ldc, stridec, group_count);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDzasum | FileCheck %s -check-prefix=cublasDzasum
// cublasDzasum: CUDA API:
// cublasDzasum-NEXT:   cublasDzasum(handle /*cublasHandle_t*/, n /*int*/,
// cublasDzasum-NEXT:                x /*const cuDoubleComplex **/, incx /*int*/, res /*double **/);
// cublasDzasum-NEXT: Is migrated to:
// cublasDzasum-NEXT:   double* res_temp_ptr_ct{{[0-9]+}} = res;
// cublasDzasum-NEXT:   if(sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasDzasum-NEXT:     res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<double>(1, dpct::get_default_queue());
// cublasDzasum-NEXT:   }
// cublasDzasum-NEXT:   oneapi::mkl::blas::column_major::asum(*handle, n, (std::complex<double>*)x, incx, res_temp_ptr_ct{{[0-9]+}});
// cublasDzasum-NEXT:   if(sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasDzasum-NEXT:     handle->wait();
// cublasDzasum-NEXT:     *res = *res_temp_ptr_ct{{[0-9]+}};
// cublasDzasum-NEXT:     sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_default_queue());
// cublasDzasum-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZgemm3m | FileCheck %s -check-prefix=cublasZgemm3m
// cublasZgemm3m: CUDA API:
// cublasZgemm3m-NEXT:   cublasZgemm3m(
// cublasZgemm3m-NEXT:       handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasZgemm3m-NEXT:       transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
// cublasZgemm3m-NEXT:       alpha /*const cuDoubleComplex **/, a /*const cuDoubleComplex **/,
// cublasZgemm3m-NEXT:       lda /*int*/, b /*const cuDoubleComplex **/, ldb /*int*/,
// cublasZgemm3m-NEXT:       beta /*const cuDoubleComplex **/, c /*cuDoubleComplex **/, ldc /*int*/);
// cublasZgemm3m-NEXT: Is migrated to:
// cublasZgemm3m-NEXT:   oneapi::mkl::blas::column_major::gemm(*handle, transa, transb, m, n, k, dpct::get_value(alpha, *handle), (std::complex<double>*)a, lda, (std::complex<double>*)b, ldb, dpct::get_value(beta, *handle), (std::complex<double>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSspmv | FileCheck %s -check-prefix=cublasSspmv
// cublasSspmv: CUDA API:
// cublasSspmv-NEXT:   cublasSspmv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasSspmv-NEXT:               n /*int*/, alpha /*const float **/, a /*const float **/,
// cublasSspmv-NEXT:               x /*const float **/, incx /*int*/, beta /*const float **/,
// cublasSspmv-NEXT:               y /*float **/, incy /*int*/);
// cublasSspmv-NEXT: Is migrated to:
// cublasSspmv-NEXT:   oneapi::mkl::blas::column_major::spmv(*handle, upper_lower, n, dpct::get_value(alpha, *handle), a, x, incx, dpct::get_value(beta, *handle), y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCsrot | FileCheck %s -check-prefix=cublasCsrot
// cublasCsrot: CUDA API:
// cublasCsrot-NEXT:   cublasCsrot(handle /*cublasHandle_t*/, n /*int*/, x /*cuComplex **/,
// cublasCsrot-NEXT:               incx /*int*/, y /*cuComplex **/, incy /*int*/,
// cublasCsrot-NEXT:               c /*const float **/, s /*const float **/);
// cublasCsrot-NEXT: Is migrated to:
// cublasCsrot-NEXT:   oneapi::mkl::blas::column_major::rot(*handle, n, (std::complex<float>*)x, incx, (std::complex<float>*)y, incy, dpct::get_value(c, *handle), dpct::get_value(s, *handle));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasGetVersion | FileCheck %s -check-prefix=cublasGetVersion
// cublasGetVersion: CUDA API:
// cublasGetVersion-NEXT:   cublasGetVersion(handle /*cublasHandle_t*/, ver /*int **/);
// cublasGetVersion-NEXT: Is migrated to:
// cublasGetVersion-NEXT:   dpct::mkl_get_version(dpct::version_field::major, ver);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasIcamin | FileCheck %s -check-prefix=cublasIcamin
// cublasIcamin: CUDA API:
// cublasIcamin-NEXT:   cublasIcamin(handle /*cublasHandle_t*/, n /*int*/, x /*const cuComplex **/,
// cublasIcamin-NEXT:                incx /*int*/, res /*int **/);
// cublasIcamin-NEXT: Is migrated to:
// cublasIcamin-NEXT:   int64_t* res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<int64_t>(1, dpct::get_default_queue());
// cublasIcamin-NEXT:   oneapi::mkl::blas::column_major::iamin(*handle, n, (std::complex<float>*)x, incx, res_temp_ptr_ct{{[0-9]+}}, oneapi::mkl::index_base::one).wait();
// cublasIcamin-NEXT:   int res_temp_host_ct{{[0-9]+}} = (int)*res_temp_ptr_ct{{[0-9]+}};
// cublasIcamin-NEXT:   dpct::dpct_memcpy(res, &res_temp_host_ct{{[0-9]+}}, sizeof(int));
// cublasIcamin-NEXT:   sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_default_queue());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasIcamax | FileCheck %s -check-prefix=cublasIcamax
// cublasIcamax: CUDA API:
// cublasIcamax-NEXT:   cublasIcamax(handle /*cublasHandle_t*/, n /*int*/, x /*const cuComplex **/,
// cublasIcamax-NEXT:                incx /*int*/, res /*int **/);
// cublasIcamax-NEXT: Is migrated to:
// cublasIcamax-NEXT:   int64_t* res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<int64_t>(1, dpct::get_default_queue());
// cublasIcamax-NEXT:   oneapi::mkl::blas::column_major::iamax(*handle, n, (std::complex<float>*)x, incx, res_temp_ptr_ct{{[0-9]+}}, oneapi::mkl::index_base::one).wait();
// cublasIcamax-NEXT:   int res_temp_host_ct{{[0-9]+}} = (int)*res_temp_ptr_ct{{[0-9]+}};
// cublasIcamax-NEXT:   dpct::dpct_memcpy(res, &res_temp_host_ct{{[0-9]+}}, sizeof(int));
// cublasIcamax-NEXT:   sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_default_queue());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDdgmm | FileCheck %s -check-prefix=cublasDdgmm
// cublasDdgmm: CUDA API:
// cublasDdgmm-NEXT:   cublasDdgmm(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasDdgmm-NEXT:               m /*int*/, n /*int*/, a /*const double **/, lda /*int*/,
// cublasDdgmm-NEXT:               x /*const double **/, incx /*int*/, c /*double **/, ldc /*int*/);
// cublasDdgmm-NEXT: Is migrated to:
// cublasDdgmm-NEXT:   oneapi::mkl::blas::column_major::dgmm_batch(*handle, left_right, m, n, a, lda, 0, x, incx, 0, c, ldc, ldc * n, 1);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasStrmv | FileCheck %s -check-prefix=cublasStrmv
// cublasStrmv: CUDA API:
// cublasStrmv-NEXT:   cublasStrmv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasStrmv-NEXT:               trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasStrmv-NEXT:               n /*int*/, a /*const float **/, lda /*int*/, x /*float **/,
// cublasStrmv-NEXT:               incx /*int*/);
// cublasStrmv-NEXT: Is migrated to:
// cublasStrmv-NEXT:   oneapi::mkl::blas::column_major::trmv(*handle, upper_lower, trans, unit_nonunit, n, a, lda, x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCrot | FileCheck %s -check-prefix=cublasCrot
// cublasCrot: CUDA API:
// cublasCrot-NEXT:   cublasCrot(handle /*cublasHandle_t*/, n /*int*/, x /*cuComplex **/,
// cublasCrot-NEXT:              incx /*int*/, y /*cuComplex **/, incy /*int*/, c /*const float **/,
// cublasCrot-NEXT:              s /*const cuComplex **/);
// cublasCrot-NEXT: Is migrated to:
// cublasCrot-NEXT:   dpct::rot(*handle, n, x, dpct::library_data_t::complex_float, incx, y, dpct::library_data_t::complex_float, incy, c, s, dpct::library_data_t::complex_float);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCreate | FileCheck %s -check-prefix=cublasCreate
// cublasCreate: CUDA API:
// cublasCreate-NEXT:   cublasCreate(handle /*cublasHandle_t **/);
// cublasCreate-NEXT: Is migrated to:
// cublasCreate-NEXT:   *handle = &dpct::get_default_queue();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasChpmv | FileCheck %s -check-prefix=cublasChpmv
// cublasChpmv: CUDA API:
// cublasChpmv-NEXT:   cublasChpmv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasChpmv-NEXT:               n /*int*/, alpha /*const cuComplex **/, a /*const cuComplex **/,
// cublasChpmv-NEXT:               x /*const cuComplex **/, incx /*int*/, beta /*const cuComplex **/,
// cublasChpmv-NEXT:               y /*cuComplex **/, incy /*int*/);
// cublasChpmv-NEXT: Is migrated to:
// cublasChpmv-NEXT:   oneapi::mkl::blas::column_major::hpmv(*handle, upper_lower, n, dpct::get_value(alpha, *handle), (std::complex<float>*)a, (std::complex<float>*)x, incx, dpct::get_value(beta, *handle), (std::complex<float>*)y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSnrm2 | FileCheck %s -check-prefix=cublasSnrm2
// cublasSnrm2: CUDA API:
// cublasSnrm2-NEXT:   cublasSnrm2(handle /*cublasHandle_t*/, n /*int*/, x /*const float **/,
// cublasSnrm2-NEXT:               incx /*int*/, res /*float **/);
// cublasSnrm2-NEXT: Is migrated to:
// cublasSnrm2-NEXT:   float* res_temp_ptr_ct{{[0-9]+}} = res;
// cublasSnrm2-NEXT:   if(sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasSnrm2-NEXT:     res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<float>(1, dpct::get_default_queue());
// cublasSnrm2-NEXT:   }
// cublasSnrm2-NEXT:   oneapi::mkl::blas::column_major::nrm2(*handle, n, x, incx, res_temp_ptr_ct{{[0-9]+}});
// cublasSnrm2-NEXT:   if(sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasSnrm2-NEXT:     handle->wait();
// cublasSnrm2-NEXT:     *res = *res_temp_ptr_ct{{[0-9]+}};
// cublasSnrm2-NEXT:     sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_default_queue());
// cublasSnrm2-NEXT:   }

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
// cublasZgemmStridedBatched-NEXT:   oneapi::mkl::blas::column_major::gemm_batch(*handle, transa, transb, m, n, k, dpct::get_value(alpha, *handle), (std::complex<double>*)a, lda, stridea, (std::complex<double>*)b, ldb, strideb, dpct::get_value(beta, *handle), (std::complex<double>*)c, ldc, stridec, group_count);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZsyr | FileCheck %s -check-prefix=cublasZsyr
// cublasZsyr: CUDA API:
// cublasZsyr-NEXT:   cublasZsyr(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZsyr-NEXT:              n /*int*/, alpha /*const cuDoubleComplex **/,
// cublasZsyr-NEXT:              x /*const cuDoubleComplex **/, incx /*int*/,
// cublasZsyr-NEXT:              a /*cuDoubleComplex **/, lda /*int*/);
// cublasZsyr-NEXT: Is migrated to:
// cublasZsyr-NEXT:   oneapi::mkl::blas::column_major::syr(*handle, upper_lower, n, dpct::get_value(alpha, *handle), (std::complex<double>*)x, incx, (std::complex<double>*)a, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDtpsv | FileCheck %s -check-prefix=cublasDtpsv
// cublasDtpsv: CUDA API:
// cublasDtpsv-NEXT:   cublasDtpsv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDtpsv-NEXT:               trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasDtpsv-NEXT:               n /*int*/, a /*const double **/, x /*double **/, incx /*int*/);
// cublasDtpsv-NEXT: Is migrated to:
// cublasDtpsv-NEXT:   oneapi::mkl::blas::column_major::tpsv(*handle, upper_lower, trans, unit_nonunit, n, a, x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDtrsv | FileCheck %s -check-prefix=cublasDtrsv
// cublasDtrsv: CUDA API:
// cublasDtrsv-NEXT:   cublasDtrsv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDtrsv-NEXT:               trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasDtrsv-NEXT:               n /*int*/, a /*const double **/, lda /*int*/, x /*double **/,
// cublasDtrsv-NEXT:               incx /*int*/);
// cublasDtrsv-NEXT: Is migrated to:
// cublasDtrsv-NEXT:   oneapi::mkl::blas::column_major::trsv(*handle, upper_lower, trans, unit_nonunit, n, a, lda, x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDgbmv | FileCheck %s -check-prefix=cublasDgbmv
// cublasDgbmv: CUDA API:
// cublasDgbmv-NEXT:   cublasDgbmv(handle /*cublasHandle_t*/, trans /*cublasOperation_t*/, m /*int*/,
// cublasDgbmv-NEXT:               n /*int*/, kl /*int*/, ku /*int*/, alpha /*const double **/,
// cublasDgbmv-NEXT:               a /*const double **/, lda /*int*/, x /*const double **/,
// cublasDgbmv-NEXT:               incx /*int*/, beta /*const double **/, y /*double **/,
// cublasDgbmv-NEXT:               incy /*int*/);
// cublasDgbmv-NEXT: Is migrated to:
// cublasDgbmv-NEXT:   oneapi::mkl::blas::column_major::gbmv(*handle, trans, m, n, kl, ku, dpct::get_value(alpha, *handle), a, lda, x, incx, dpct::get_value(beta, *handle), y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZgerc | FileCheck %s -check-prefix=cublasZgerc
// cublasZgerc: CUDA API:
// cublasZgerc-NEXT:   cublasZgerc(handle /*cublasHandle_t*/, m /*int*/, n /*int*/,
// cublasZgerc-NEXT:               alpha /*const cuDoubleComplex **/, x /*const cuDoubleComplex **/,
// cublasZgerc-NEXT:               incx /*int*/, y /*const cuDoubleComplex **/, incy /*int*/,
// cublasZgerc-NEXT:               a /*cuDoubleComplex **/, lda /*int*/);
// cublasZgerc-NEXT: Is migrated to:
// cublasZgerc-NEXT:   oneapi::mkl::blas::column_major::gerc(*handle, m, n, dpct::get_value(alpha, *handle), (std::complex<double>*)x, incx, (std::complex<double>*)y, incy, (std::complex<double>*)a, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasStbsv | FileCheck %s -check-prefix=cublasStbsv
// cublasStbsv: CUDA API:
// cublasStbsv-NEXT:   cublasStbsv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasStbsv-NEXT:               trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasStbsv-NEXT:               n /*int*/, k /*int*/, a /*const float **/, lda /*int*/,
// cublasStbsv-NEXT:               x /*float **/, incx /*int*/);
// cublasStbsv-NEXT: Is migrated to:
// cublasStbsv-NEXT:   oneapi::mkl::blas::column_major::tbsv(*handle, upper_lower, trans, unit_nonunit, n, k, a, lda, x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasStrsm | FileCheck %s -check-prefix=cublasStrsm
// cublasStrsm: CUDA API:
// cublasStrsm-NEXT:   cublasStrsm(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasStrsm-NEXT:               upper_lower /*cublasFillMode_t*/, transa /*cublasOperation_t*/,
// cublasStrsm-NEXT:               unit_diag /*cublasDiagType_t*/, m /*int*/, n /*int*/,
// cublasStrsm-NEXT:               alpha /*const float **/, a /*const float **/, lda /*int*/,
// cublasStrsm-NEXT:               b /*float **/, ldb /*int*/);
// cublasStrsm-NEXT: Is migrated to:
// cublasStrsm-NEXT:   oneapi::mkl::blas::column_major::trsm(*handle, left_right, upper_lower, transa, unit_diag, m, n, dpct::get_value(alpha, *handle), a, lda, b, ldb);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCdotc | FileCheck %s -check-prefix=cublasCdotc
// cublasCdotc: CUDA API:
// cublasCdotc-NEXT:   cublasCdotc(handle /*cublasHandle_t*/, n /*int*/, x /*const cuComplex **/,
// cublasCdotc-NEXT:               incx /*int*/, y /*const cuComplex **/, incy /*int*/,
// cublasCdotc-NEXT:               res /*cuComplex **/);
// cublasCdotc-NEXT: Is migrated to:
// cublasCdotc-NEXT:   sycl::float2* res_temp_ptr_ct{{[0-9]+}} = res;
// cublasCdotc-NEXT:   if(sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasCdotc-NEXT:     res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<sycl::float2>(1, dpct::get_default_queue());
// cublasCdotc-NEXT:   }
// cublasCdotc-NEXT:   oneapi::mkl::blas::column_major::dotc(*handle, n, (std::complex<float>*)x, incx, (std::complex<float>*)y, incy, (std::complex<float>*)res_temp_ptr_ct{{[0-9]+}});
// cublasCdotc-NEXT:   if(sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasCdotc-NEXT:     handle->wait();
// cublasCdotc-NEXT:     *res = *res_temp_ptr_ct{{[0-9]+}};
// cublasCdotc-NEXT:     sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_default_queue());
// cublasCdotc-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSgeam | FileCheck %s -check-prefix=cublasSgeam
// cublasSgeam: CUDA API:
// cublasSgeam-NEXT:   cublasSgeam(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasSgeam-NEXT:               transb /*cublasOperation_t*/, m /*int*/, n /*int*/,
// cublasSgeam-NEXT:               alpha /*const float **/, a /*const float **/, lda /*int*/,
// cublasSgeam-NEXT:               beta /*const float **/, b /*const float **/, ldb /*int*/,
// cublasSgeam-NEXT:               c /*float **/, ldc /*int*/);
// cublasSgeam-NEXT: Is migrated to:
// cublasSgeam-NEXT:   oneapi::mkl::blas::column_major::omatadd(*handle, transa, transb, m, n, dpct::get_value(alpha, *handle), a, lda, dpct::get_value(beta, *handle), b, ldb, c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCgeru | FileCheck %s -check-prefix=cublasCgeru
// cublasCgeru: CUDA API:
// cublasCgeru-NEXT:   cublasCgeru(handle /*cublasHandle_t*/, m /*int*/, n /*int*/,
// cublasCgeru-NEXT:               alpha /*const cuComplex **/, x /*const cuComplex **/,
// cublasCgeru-NEXT:               incx /*int*/, y /*const cuComplex **/, incy /*int*/,
// cublasCgeru-NEXT:               a /*cuComplex **/, lda /*int*/);
// cublasCgeru-NEXT: Is migrated to:
// cublasCgeru-NEXT:   oneapi::mkl::blas::column_major::geru(*handle, m, n, dpct::get_value(alpha, *handle), (std::complex<float>*)x, incx, (std::complex<float>*)y, incy, (std::complex<float>*)a, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCgemm | FileCheck %s -check-prefix=cublasCgemm
// cublasCgemm: CUDA API:
// cublasCgemm-NEXT:   cublasCgemm(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasCgemm-NEXT:               transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
// cublasCgemm-NEXT:               alpha /*const cuComplex **/, a /*const cuComplex **/, lda /*int*/,
// cublasCgemm-NEXT:               b /*const cuComplex **/, ldb /*int*/, beta /*const cuComplex **/,
// cublasCgemm-NEXT:               c /*cuComplex **/, ldc /*int*/);
// cublasCgemm-NEXT: Is migrated to:
// cublasCgemm-NEXT:   oneapi::mkl::blas::column_major::gemm(*handle, transa, transb, m, n, k, dpct::get_value(alpha, *handle), (std::complex<float>*)a, lda, (std::complex<float>*)b, ldb, dpct::get_value(beta, *handle), (std::complex<float>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCgemm3m | FileCheck %s -check-prefix=cublasCgemm3m
// cublasCgemm3m: CUDA API:
// cublasCgemm3m-NEXT:   cublasCgemm3m(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasCgemm3m-NEXT:                 transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
// cublasCgemm3m-NEXT:                 alpha /*const cuComplex **/, a /*const cuComplex **/,
// cublasCgemm3m-NEXT:                 lda /*int*/, b /*const cuComplex **/, ldb /*int*/,
// cublasCgemm3m-NEXT:                 beta /*const cuComplex **/, c /*cuComplex **/, ldc /*int*/);
// cublasCgemm3m-NEXT: Is migrated to:
// cublasCgemm3m-NEXT:   oneapi::mkl::blas::column_major::gemm(*handle, transa, transb, m, n, k, dpct::get_value(alpha, *handle), (std::complex<float>*)a, lda, (std::complex<float>*)b, ldb, dpct::get_value(beta, *handle), (std::complex<float>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDtbsv | FileCheck %s -check-prefix=cublasDtbsv
// cublasDtbsv: CUDA API:
// cublasDtbsv-NEXT:   cublasDtbsv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDtbsv-NEXT:               trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasDtbsv-NEXT:               n /*int*/, k /*int*/, a /*const double **/, lda /*int*/,
// cublasDtbsv-NEXT:               x /*double **/, incx /*int*/);
// cublasDtbsv-NEXT: Is migrated to:
// cublasDtbsv-NEXT:   oneapi::mkl::blas::column_major::tbsv(*handle, upper_lower, trans, unit_nonunit, n, k, a, lda, x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZaxpy | FileCheck %s -check-prefix=cublasZaxpy
// cublasZaxpy: CUDA API:
// cublasZaxpy-NEXT:   cublasZaxpy(handle /*cublasHandle_t*/, n /*int*/,
// cublasZaxpy-NEXT:               alpha /*const cuDoubleComplex **/, x /*const cuDoubleComplex **/,
// cublasZaxpy-NEXT:               incx /*int*/, y /*cuDoubleComplex **/, incy /*int*/);
// cublasZaxpy-NEXT: Is migrated to:
// cublasZaxpy-NEXT:   oneapi::mkl::blas::column_major::axpy(*handle, n, dpct::get_value(alpha, *handle), (std::complex<double>*)x, incx, (std::complex<double>*)y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZsymv | FileCheck %s -check-prefix=cublasZsymv
// cublasZsymv: CUDA API:
// cublasZsymv-NEXT:   cublasZsymv(
// cublasZsymv-NEXT:       handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/, n /*int*/,
// cublasZsymv-NEXT:       alpha /*const cuDoubleComplex **/, a /*const cuDoubleComplex **/,
// cublasZsymv-NEXT:       lda /*int*/, x /*const cuDoubleComplex **/, incx /*int*/,
// cublasZsymv-NEXT:       beta /*const cuDoubleComplex **/, y /*cuDoubleComplex **/, incy /*int*/);
// cublasZsymv-NEXT: Is migrated to:
// cublasZsymv-NEXT:   oneapi::mkl::blas::column_major::symv(*handle, upper_lower, n, dpct::get_value(alpha, *handle), (std::complex<double>*)a, lda, (std::complex<double>*)x, incx, dpct::get_value(beta, *handle), (std::complex<double>*)y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSswap | FileCheck %s -check-prefix=cublasSswap
// cublasSswap: CUDA API:
// cublasSswap-NEXT:   cublasSswap(handle /*cublasHandle_t*/, n /*int*/, x /*float **/, incx /*int*/,
// cublasSswap-NEXT:               y /*float **/, incy /*int*/);
// cublasSswap-NEXT: Is migrated to:
// cublasSswap-NEXT:   oneapi::mkl::blas::column_major::swap(*handle, n, x, incx, y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZtrmm | FileCheck %s -check-prefix=cublasZtrmm
// cublasZtrmm: CUDA API:
// cublasZtrmm-NEXT:   cublasZtrmm(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasZtrmm-NEXT:               upper_lower /*cublasFillMode_t*/, transa /*cublasOperation_t*/,
// cublasZtrmm-NEXT:               unit_diag /*cublasDiagType_t*/, m /*int*/, n /*int*/,
// cublasZtrmm-NEXT:               alpha /*const cuDoubleComplex **/, a /*const cuDoubleComplex **/,
// cublasZtrmm-NEXT:               lda /*int*/, b /*const cuDoubleComplex **/, ldb /*int*/,
// cublasZtrmm-NEXT:               c /*cuDoubleComplex **/, ldc /*int*/);
// cublasZtrmm-NEXT: Is migrated to:
// cublasZtrmm-NEXT:   dpct::trmm(*handle, left_right, upper_lower, transa, unit_diag, m, n, alpha, a, lda, b, ldb, c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasStrmm | FileCheck %s -check-prefix=cublasStrmm
// cublasStrmm: CUDA API:
// cublasStrmm-NEXT:   cublasStrmm(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasStrmm-NEXT:               upper_lower /*cublasFillMode_t*/, transa /*cublasOperation_t*/,
// cublasStrmm-NEXT:               unit_diag /*cublasDiagType_t*/, m /*int*/, n /*int*/,
// cublasStrmm-NEXT:               alpha /*const float **/, a /*const float **/, lda /*int*/,
// cublasStrmm-NEXT:               b /*const float **/, ldb /*int*/, c /*float **/, ldc /*int*/);
// cublasStrmm-NEXT: Is migrated to:
// cublasStrmm-NEXT:   dpct::trmm(*handle, left_right, upper_lower, transa, unit_diag, m, n, alpha, a, lda, b, ldb, c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDnrm2 | FileCheck %s -check-prefix=cublasDnrm2
// cublasDnrm2: CUDA API:
// cublasDnrm2-NEXT:   cublasDnrm2(handle /*cublasHandle_t*/, n /*int*/, x /*const double **/,
// cublasDnrm2-NEXT:               incx /*int*/, res /*double **/);
// cublasDnrm2-NEXT: Is migrated to:
// cublasDnrm2-NEXT:   double* res_temp_ptr_ct{{[0-9]+}} = res;
// cublasDnrm2-NEXT:   if(sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasDnrm2-NEXT:     res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<double>(1, dpct::get_default_queue());
// cublasDnrm2-NEXT:   }
// cublasDnrm2-NEXT:   oneapi::mkl::blas::column_major::nrm2(*handle, n, x, incx, res_temp_ptr_ct{{[0-9]+}});
// cublasDnrm2-NEXT:   if(sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasDnrm2-NEXT:     handle->wait();
// cublasDnrm2-NEXT:     *res = *res_temp_ptr_ct{{[0-9]+}};
// cublasDnrm2-NEXT:     sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_default_queue());
// cublasDnrm2-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDrot | FileCheck %s -check-prefix=cublasDrot
// cublasDrot: CUDA API:
// cublasDrot-NEXT:   cublasDrot(handle /*cublasHandle_t*/, n /*int*/, x /*double **/, incx /*int*/,
// cublasDrot-NEXT:              y /*double **/, incy /*int*/, c /*const double **/,
// cublasDrot-NEXT:              s /*const double **/);
// cublasDrot-NEXT: Is migrated to:
// cublasDrot-NEXT:   oneapi::mkl::blas::column_major::rot(*handle, n, x, incx, y, incy, dpct::get_value(c, *handle), dpct::get_value(s, *handle));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDznrm2 | FileCheck %s -check-prefix=cublasDznrm2
// cublasDznrm2: CUDA API:
// cublasDznrm2-NEXT:   cublasDznrm2(handle /*cublasHandle_t*/, n /*int*/,
// cublasDznrm2-NEXT:                x /*const cuDoubleComplex **/, incx /*int*/, res /*double **/);
// cublasDznrm2-NEXT: Is migrated to:
// cublasDznrm2-NEXT:   double* res_temp_ptr_ct{{[0-9]+}} = res;
// cublasDznrm2-NEXT:   if(sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasDznrm2-NEXT:     res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<double>(1, dpct::get_default_queue());
// cublasDznrm2-NEXT:   }
// cublasDznrm2-NEXT:   oneapi::mkl::blas::column_major::nrm2(*handle, n, (std::complex<double>*)x, incx, res_temp_ptr_ct{{[0-9]+}});
// cublasDznrm2-NEXT:   if(sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasDznrm2-NEXT:     handle->wait();
// cublasDznrm2-NEXT:     *res = *res_temp_ptr_ct{{[0-9]+}};
// cublasDznrm2-NEXT:     sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_default_queue());
// cublasDznrm2-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSgbmv | FileCheck %s -check-prefix=cublasSgbmv
// cublasSgbmv: CUDA API:
// cublasSgbmv-NEXT:   cublasSgbmv(handle /*cublasHandle_t*/, trans /*cublasOperation_t*/, m /*int*/,
// cublasSgbmv-NEXT:               n /*int*/, kl /*int*/, ku /*int*/, alpha /*const float **/,
// cublasSgbmv-NEXT:               a /*const float **/, lda /*int*/, x /*const float **/,
// cublasSgbmv-NEXT:               incx /*int*/, beta /*const float **/, y /*float **/,
// cublasSgbmv-NEXT:               incy /*int*/);
// cublasSgbmv-NEXT: Is migrated to:
// cublasSgbmv-NEXT:   oneapi::mkl::blas::column_major::gbmv(*handle, trans, m, n, kl, ku, dpct::get_value(alpha, *handle), a, lda, x, incx, dpct::get_value(beta, *handle), y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSetVectorAsync | FileCheck %s -check-prefix=cublasSetVectorAsync
// cublasSetVectorAsync: CUDA API:
// cublasSetVectorAsync-NEXT:   cublasSetVectorAsync(n /*int*/, elementsize /*int*/, from /*const void **/,
// cublasSetVectorAsync-NEXT:                        incx /*int*/, to /*void **/, incy /*int*/,
// cublasSetVectorAsync-NEXT:                        stream /*cudaStream_t*/);
// cublasSetVectorAsync-NEXT: Is migrated to:
// cublasSetVectorAsync-NEXT:   dpct::matrix_mem_copy((void*)to, (void*)from, incy, incx, 1, n, elementsize, dpct::automatic, *stream, true);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSsymv | FileCheck %s -check-prefix=cublasSsymv
// cublasSsymv: CUDA API:
// cublasSsymv-NEXT:   cublasSsymv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasSsymv-NEXT:               n /*int*/, alpha /*const float **/, a /*const float **/,
// cublasSsymv-NEXT:               lda /*int*/, x /*const float **/, incx /*int*/,
// cublasSsymv-NEXT:               beta /*const float **/, y /*float **/, incy /*int*/);
// cublasSsymv-NEXT: Is migrated to:
// cublasSsymv-NEXT:   oneapi::mkl::blas::column_major::symv(*handle, upper_lower, n, dpct::get_value(alpha, *handle), a, lda, x, incx, dpct::get_value(beta, *handle), y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCtrsv | FileCheck %s -check-prefix=cublasCtrsv
// cublasCtrsv: CUDA API:
// cublasCtrsv-NEXT:   cublasCtrsv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCtrsv-NEXT:               trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasCtrsv-NEXT:               n /*int*/, a /*const cuComplex **/, lda /*int*/,
// cublasCtrsv-NEXT:               x /*cuComplex **/, incx /*int*/);
// cublasCtrsv-NEXT: Is migrated to:
// cublasCtrsv-NEXT:   oneapi::mkl::blas::column_major::trsv(*handle, upper_lower, trans, unit_nonunit, n, (std::complex<float>*)a, lda, (std::complex<float>*)x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDasum | FileCheck %s -check-prefix=cublasDasum
// cublasDasum: CUDA API:
// cublasDasum-NEXT:   cublasDasum(handle /*cublasHandle_t*/, n /*int*/, x /*const double **/,
// cublasDasum-NEXT:               incx /*int*/, res /*double **/);
// cublasDasum-NEXT: Is migrated to:
// cublasDasum-NEXT:   double* res_temp_ptr_ct{{[0-9]+}} = res;
// cublasDasum-NEXT:   if(sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasDasum-NEXT:     res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<double>(1, dpct::get_default_queue());
// cublasDasum-NEXT:   }
// cublasDasum-NEXT:   oneapi::mkl::blas::column_major::asum(*handle, n, x, incx, res_temp_ptr_ct{{[0-9]+}});
// cublasDasum-NEXT:   if(sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasDasum-NEXT:     handle->wait();
// cublasDasum-NEXT:     *res = *res_temp_ptr_ct{{[0-9]+}};
// cublasDasum-NEXT:     sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_default_queue());
// cublasDasum-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSsbmv | FileCheck %s -check-prefix=cublasSsbmv
// cublasSsbmv: CUDA API:
// cublasSsbmv-NEXT:   cublasSsbmv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasSsbmv-NEXT:               n /*int*/, k /*int*/, alpha /*const float **/,
// cublasSsbmv-NEXT:               a /*const float **/, lda /*int*/, x /*const float **/,
// cublasSsbmv-NEXT:               incx /*int*/, beta /*const float **/, y /*float **/,
// cublasSsbmv-NEXT:               incy /*int*/);
// cublasSsbmv-NEXT: Is migrated to:
// cublasSsbmv-NEXT:   oneapi::mkl::blas::column_major::sbmv(*handle, upper_lower, n, k, dpct::get_value(alpha, *handle), a, lda, x, incx, dpct::get_value(beta, *handle), y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasIzamin | FileCheck %s -check-prefix=cublasIzamin
// cublasIzamin: CUDA API:
// cublasIzamin-NEXT:   cublasIzamin(handle /*cublasHandle_t*/, n /*int*/,
// cublasIzamin-NEXT:                x /*const cuDoubleComplex **/, incx /*int*/, res /*int **/);
// cublasIzamin-NEXT: Is migrated to:
// cublasIzamin-NEXT:   int64_t* res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<int64_t>(1, dpct::get_default_queue());
// cublasIzamin-NEXT:   oneapi::mkl::blas::column_major::iamin(*handle, n, (std::complex<double>*)x, incx, res_temp_ptr_ct{{[0-9]+}}, oneapi::mkl::index_base::one).wait();
// cublasIzamin-NEXT:   int res_temp_host_ct{{[0-9]+}} = (int)*res_temp_ptr_ct{{[0-9]+}};
// cublasIzamin-NEXT:   dpct::dpct_memcpy(res, &res_temp_host_ct{{[0-9]+}}, sizeof(int));
// cublasIzamin-NEXT:   sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_default_queue());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasNrm2Ex | FileCheck %s -check-prefix=cublasNrm2Ex
// cublasNrm2Ex: CUDA API:
// cublasNrm2Ex-NEXT:   cublasNrm2Ex(handle /*cublasHandle_t*/, n /*int*/, x /*const void **/,
// cublasNrm2Ex-NEXT:                xtype /*cudaDataType*/, incx /*int*/, res /*void **/,
// cublasNrm2Ex-NEXT:                restype /*cudaDataType*/, computetype /*cudaDataType*/);
// cublasNrm2Ex-NEXT: Is migrated to:
// cublasNrm2Ex-NEXT:   dpct::nrm2(*handle, n, x, xtype, incx, res, restype);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDsyr | FileCheck %s -check-prefix=cublasDsyr
// cublasDsyr: CUDA API:
// cublasDsyr-NEXT:   cublasDsyr(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDsyr-NEXT:              n /*int*/, alpha /*const double **/, x /*const double **/,
// cublasDsyr-NEXT:              incx /*int*/, a /*double **/, lda /*int*/);
// cublasDsyr-NEXT: Is migrated to:
// cublasDsyr-NEXT:   oneapi::mkl::blas::column_major::syr(*handle, upper_lower, n, dpct::get_value(alpha, *handle), x, incx, a, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZhpr2 | FileCheck %s -check-prefix=cublasZhpr2
// cublasZhpr2: CUDA API:
// cublasZhpr2-NEXT:   cublasZhpr2(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZhpr2-NEXT:               n /*int*/, alpha /*const cuDoubleComplex **/,
// cublasZhpr2-NEXT:               x /*const cuDoubleComplex **/, incx /*int*/,
// cublasZhpr2-NEXT:               y /*const cuDoubleComplex **/, incy /*int*/,
// cublasZhpr2-NEXT:               a /*cuDoubleComplex **/);
// cublasZhpr2-NEXT: Is migrated to:
// cublasZhpr2-NEXT:   oneapi::mkl::blas::column_major::hpr2(*handle, upper_lower, n, dpct::get_value(alpha, *handle), (std::complex<double>*)x, incx, (std::complex<double>*)y, incy, (std::complex<double>*)a);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZtpmv | FileCheck %s -check-prefix=cublasZtpmv
// cublasZtpmv: CUDA API:
// cublasZtpmv-NEXT:   cublasZtpmv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZtpmv-NEXT:               trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasZtpmv-NEXT:               n /*int*/, a /*const cuDoubleComplex **/, x /*cuDoubleComplex **/,
// cublasZtpmv-NEXT:               incx /*int*/);
// cublasZtpmv-NEXT: Is migrated to:
// cublasZtpmv-NEXT:   oneapi::mkl::blas::column_major::tpmv(*handle, upper_lower, trans, unit_nonunit, n, (std::complex<double>*)a, (std::complex<double>*)x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZherkx | FileCheck %s -check-prefix=cublasZherkx
// cublasZherkx: CUDA API:
// cublasZherkx-NEXT:   cublasZherkx(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZherkx-NEXT:                trans /*cublasOperation_t*/, n /*int*/, k /*int*/,
// cublasZherkx-NEXT:                alpha /*const cuDoubleComplex **/, a /*const cuDoubleComplex **/,
// cublasZherkx-NEXT:                lda /*int*/, b /*const cuDoubleComplex **/, ldb /*int*/,
// cublasZherkx-NEXT:                beta /*const double **/, c /*cuDoubleComplex **/, ldc /*int*/);
// cublasZherkx-NEXT: Is migrated to:
// cublasZherkx-NEXT:   dpct::herk(*handle, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasChbmv | FileCheck %s -check-prefix=cublasChbmv
// cublasChbmv: CUDA API:
// cublasChbmv-NEXT:   cublasChbmv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasChbmv-NEXT:               n /*int*/, k /*int*/, alpha /*const cuComplex **/,
// cublasChbmv-NEXT:               a /*const cuComplex **/, lda /*int*/, x /*const cuComplex **/,
// cublasChbmv-NEXT:               incx /*int*/, beta /*const cuComplex **/, y /*cuComplex **/,
// cublasChbmv-NEXT:               incy /*int*/);
// cublasChbmv-NEXT: Is migrated to:
// cublasChbmv-NEXT:   oneapi::mkl::blas::column_major::hbmv(*handle, upper_lower, n, k, dpct::get_value(alpha, *handle), (std::complex<float>*)a, lda, (std::complex<float>*)x, incx, dpct::get_value(beta, *handle), (std::complex<float>*)y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSrotm | FileCheck %s -check-prefix=cublasSrotm
// cublasSrotm: CUDA API:
// cublasSrotm-NEXT:   cublasSrotm(handle /*cublasHandle_t*/, n /*int*/, x /*float **/, incx /*int*/,
// cublasSrotm-NEXT:               y /*float **/, incy /*int*/, param /*const float **/);
// cublasSrotm-NEXT: Is migrated to:
// cublasSrotm-NEXT:   oneapi::mkl::blas::column_major::rotm(*handle, n, x, incx, y, incy, const_cast<float*>(param));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCher | FileCheck %s -check-prefix=cublasCher
// cublasCher: CUDA API:
// cublasCher-NEXT:   cublasCher(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCher-NEXT:              n /*int*/, alpha /*const float **/, x /*const cuComplex **/,
// cublasCher-NEXT:              incx /*int*/, a /*cuComplex **/, lda /*int*/);
// cublasCher-NEXT: Is migrated to:
// cublasCher-NEXT:   oneapi::mkl::blas::column_major::her(*handle, upper_lower, n, dpct::get_value(alpha, *handle), (std::complex<float>*)x, incx, (std::complex<float>*)a, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSgemm | FileCheck %s -check-prefix=cublasSgemm
// cublasSgemm: CUDA API:
// cublasSgemm-NEXT:   cublasSgemm(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasSgemm-NEXT:               transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
// cublasSgemm-NEXT:               alpha /*const float **/, a /*const float **/, lda /*int*/,
// cublasSgemm-NEXT:               b /*const float **/, ldb /*int*/, beta /*const float **/,
// cublasSgemm-NEXT:               c /*float **/, ldc /*int*/);
// cublasSgemm-NEXT: Is migrated to:
// cublasSgemm-NEXT:   oneapi::mkl::blas::column_major::gemm(*handle, transa, transb, m, n, k, dpct::get_value(alpha, *handle), a, lda, b, ldb, dpct::get_value(beta, *handle), c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSsyrk | FileCheck %s -check-prefix=cublasSsyrk
// cublasSsyrk: CUDA API:
// cublasSsyrk-NEXT:   cublasSsyrk(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasSsyrk-NEXT:               trans /*cublasOperation_t*/, n /*int*/, k /*int*/,
// cublasSsyrk-NEXT:               alpha /*const float **/, a /*const float **/, lda /*int*/,
// cublasSsyrk-NEXT:               beta /*const float **/, c /*float **/, ldc /*int*/);
// cublasSsyrk-NEXT: Is migrated to:
// cublasSsyrk-NEXT:   oneapi::mkl::blas::column_major::syrk(*handle, upper_lower, trans, n, k, dpct::get_value(alpha, *handle), a, lda, dpct::get_value(beta, *handle), c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZhemm | FileCheck %s -check-prefix=cublasZhemm
// cublasZhemm: CUDA API:
// cublasZhemm-NEXT:   cublasZhemm(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasZhemm-NEXT:               upper_lower /*cublasFillMode_t*/, m /*int*/, n /*int*/,
// cublasZhemm-NEXT:               alpha /*const cuDoubleComplex **/, a /*const cuDoubleComplex **/,
// cublasZhemm-NEXT:               lda /*int*/, b /*const cuDoubleComplex **/, ldb /*int*/,
// cublasZhemm-NEXT:               beta /*const cuDoubleComplex **/, c /*cuDoubleComplex **/,
// cublasZhemm-NEXT:               ldc /*int*/);
// cublasZhemm-NEXT: Is migrated to:
// cublasZhemm-NEXT:   oneapi::mkl::blas::column_major::hemm(*handle, left_right, upper_lower, m, n, dpct::get_value(alpha, *handle), (std::complex<double>*)a, lda, (std::complex<double>*)b, ldb, dpct::get_value(beta, *handle), (std::complex<double>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSetMatrixAsync | FileCheck %s -check-prefix=cublasSetMatrixAsync
// cublasSetMatrixAsync: CUDA API:
// cublasSetMatrixAsync-NEXT:   cublasSetMatrixAsync(rows /*int*/, cols /*int*/, elementsize /*int*/,
// cublasSetMatrixAsync-NEXT:                        a /*const void **/, lda /*int*/, b /*void **/,
// cublasSetMatrixAsync-NEXT:                        ldb /*int*/, stream /*cudaStream_t*/);
// cublasSetMatrixAsync-NEXT: Is migrated to:
// cublasSetMatrixAsync-NEXT:   dpct::matrix_mem_copy((void*)b, (void*)a, ldb, lda, rows, cols, elementsize, dpct::automatic, *stream, true);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSaxpy | FileCheck %s -check-prefix=cublasSaxpy
// cublasSaxpy: CUDA API:
// cublasSaxpy-NEXT:   cublasSaxpy(handle /*cublasHandle_t*/, n /*int*/, alpha /*const float **/,
// cublasSaxpy-NEXT:               x /*const float **/, incx /*int*/, y /*float **/, incy /*int*/);
// cublasSaxpy-NEXT: Is migrated to:
// cublasSaxpy-NEXT:   oneapi::mkl::blas::column_major::axpy(*handle, n, dpct::get_value(alpha, *handle), x, incx, y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSetAtomicsMode | FileCheck %s -check-prefix=cublasSetAtomicsMode
// cublasSetAtomicsMode: CUDA API:
// cublasSetAtomicsMode-NEXT:   cublasSetAtomicsMode(handle /*cublasHandle_t*/,
// cublasSetAtomicsMode-NEXT:                        atomics /*cublasAtomicsMode_t*/);
// cublasSetAtomicsMode-NEXT: Is migrated to:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDsymv | FileCheck %s -check-prefix=cublasDsymv
// cublasDsymv: CUDA API:
// cublasDsymv-NEXT:   cublasDsymv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDsymv-NEXT:               n /*int*/, alpha /*const double **/, a /*const double **/,
// cublasDsymv-NEXT:               lda /*int*/, x /*const double **/, incx /*int*/,
// cublasDsymv-NEXT:               beta /*const double **/, y /*double **/, incy /*int*/);
// cublasDsymv-NEXT: Is migrated to:
// cublasDsymv-NEXT:   oneapi::mkl::blas::column_major::symv(*handle, upper_lower, n, dpct::get_value(alpha, *handle), a, lda, x, incx, dpct::get_value(beta, *handle), y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDsymm | FileCheck %s -check-prefix=cublasDsymm
// cublasDsymm: CUDA API:
// cublasDsymm-NEXT:   cublasDsymm(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasDsymm-NEXT:               upper_lower /*cublasFillMode_t*/, m /*int*/, n /*int*/,
// cublasDsymm-NEXT:               alpha /*const double **/, a /*const double **/, lda /*int*/,
// cublasDsymm-NEXT:               b /*const double **/, ldb /*int*/, beta /*const double **/,
// cublasDsymm-NEXT:               c /*double **/, ldc /*int*/);
// cublasDsymm-NEXT: Is migrated to:
// cublasDsymm-NEXT:   oneapi::mkl::blas::column_major::symm(*handle, left_right, upper_lower, m, n, dpct::get_value(alpha, *handle), a, lda, b, ldb, dpct::get_value(beta, *handle), c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasChpr2 | FileCheck %s -check-prefix=cublasChpr2
// cublasChpr2: CUDA API:
// cublasChpr2-NEXT:   cublasChpr2(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasChpr2-NEXT:               n /*int*/, alpha /*const cuComplex **/, x /*const cuComplex **/,
// cublasChpr2-NEXT:               incx /*int*/, y /*const cuComplex **/, incy /*int*/,
// cublasChpr2-NEXT:               a /*cuComplex **/);
// cublasChpr2-NEXT: Is migrated to:
// cublasChpr2-NEXT:   oneapi::mkl::blas::column_major::hpr2(*handle, upper_lower, n, dpct::get_value(alpha, *handle), (std::complex<float>*)x, incx, (std::complex<float>*)y, incy, (std::complex<float>*)a);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCtbsv | FileCheck %s -check-prefix=cublasCtbsv
// cublasCtbsv: CUDA API:
// cublasCtbsv-NEXT:   cublasCtbsv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCtbsv-NEXT:               trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasCtbsv-NEXT:               n /*int*/, k /*int*/, a /*const cuComplex **/, lda /*int*/,
// cublasCtbsv-NEXT:               x /*cuComplex **/, incx /*int*/);
// cublasCtbsv-NEXT: Is migrated to:
// cublasCtbsv-NEXT:   oneapi::mkl::blas::column_major::tbsv(*handle, upper_lower, trans, unit_nonunit, n, k, (std::complex<float>*)a, lda, (std::complex<float>*)x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDsbmv | FileCheck %s -check-prefix=cublasDsbmv
// cublasDsbmv: CUDA API:
// cublasDsbmv-NEXT:   cublasDsbmv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDsbmv-NEXT:               n /*int*/, k /*int*/, alpha /*const double **/,
// cublasDsbmv-NEXT:               a /*const double **/, lda /*int*/, x /*const double **/,
// cublasDsbmv-NEXT:               incx /*int*/, beta /*const double **/, y /*double **/,
// cublasDsbmv-NEXT:               incy /*int*/);
// cublasDsbmv-NEXT: Is migrated to:
// cublasDsbmv-NEXT:   oneapi::mkl::blas::column_major::sbmv(*handle, upper_lower, n, k, dpct::get_value(alpha, *handle), a, lda, x, incx, dpct::get_value(beta, *handle), y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCgemmEx | FileCheck %s -check-prefix=cublasCgemmEx
// cublasCgemmEx: CUDA API:
// cublasCgemmEx-NEXT:   cublasCgemmEx(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasCgemmEx-NEXT:                 transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
// cublasCgemmEx-NEXT:                 alpha /*const cuComplex **/, a /*const void **/,
// cublasCgemmEx-NEXT:                 atype /*cudaDataType*/, lda /*int*/, b /*const void **/,
// cublasCgemmEx-NEXT:                 btype /*cudaDataType*/, ldb /*int*/, beta /*const cuComplex **/,
// cublasCgemmEx-NEXT:                 c /*void **/, ctype /*cudaDataType*/, ldc /*int*/);
// cublasCgemmEx-NEXT: Is migrated to:
// cublasCgemmEx-NEXT:   dpct::gemm(*handle, transa, transb, m, n, k, alpha, a, atype, lda, b, btype, ldb, beta, c, ctype, ldc, dpct::library_data_t::complex_float);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasIsamax | FileCheck %s -check-prefix=cublasIsamax
// cublasIsamax: CUDA API:
// cublasIsamax-NEXT:   cublasIsamax(handle /*cublasHandle_t*/, n /*int*/, x /*const float **/,
// cublasIsamax-NEXT:                incx /*int*/, res /*int **/);
// cublasIsamax-NEXT: Is migrated to:
// cublasIsamax-NEXT:   int64_t* res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<int64_t>(1, dpct::get_default_queue());
// cublasIsamax-NEXT:   oneapi::mkl::blas::column_major::iamax(*handle, n, x, incx, res_temp_ptr_ct{{[0-9]+}}, oneapi::mkl::index_base::one).wait();
// cublasIsamax-NEXT:   int res_temp_host_ct{{[0-9]+}} = (int)*res_temp_ptr_ct{{[0-9]+}};
// cublasIsamax-NEXT:   dpct::dpct_memcpy(res, &res_temp_host_ct{{[0-9]+}}, sizeof(int));
// cublasIsamax-NEXT:   sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_default_queue());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasGetMatrix | FileCheck %s -check-prefix=cublasGetMatrix
// cublasGetMatrix: CUDA API:
// cublasGetMatrix-NEXT:   cublasGetMatrix(rows /*int*/, cols /*int*/, elementsize /*int*/,
// cublasGetMatrix-NEXT:                   a /*const void **/, lda /*int*/, b /*void **/, ldb /*int*/);
// cublasGetMatrix-NEXT: Is migrated to:
// cublasGetMatrix-NEXT:   dpct::matrix_mem_copy((void*)b, (void*)a, ldb, lda, rows, cols, elementsize);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZgeam | FileCheck %s -check-prefix=cublasZgeam
// cublasZgeam: CUDA API:
// cublasZgeam-NEXT:   cublasZgeam(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasZgeam-NEXT:               transb /*cublasOperation_t*/, m /*int*/, n /*int*/,
// cublasZgeam-NEXT:               alpha /*const cuDoubleComplex **/, a /*const cuDoubleComplex **/,
// cublasZgeam-NEXT:               lda /*int*/, beta /*const cuDoubleComplex **/,
// cublasZgeam-NEXT:               b /*const cuDoubleComplex **/, ldb /*int*/,
// cublasZgeam-NEXT:               c /*cuDoubleComplex **/, ldc /*int*/);
// cublasZgeam-NEXT: Is migrated to:
// cublasZgeam-NEXT:   oneapi::mkl::blas::column_major::omatadd(*handle, transa, transb, m, n, dpct::get_value(alpha, *handle), (std::complex<double>*)a, lda, dpct::get_value(beta, *handle), (std::complex<double>*)b, ldb, (std::complex<double>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCsyr2 | FileCheck %s -check-prefix=cublasCsyr2
// cublasCsyr2: CUDA API:
// cublasCsyr2-NEXT:   cublasCsyr2(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCsyr2-NEXT:               n /*int*/, alpha /*const cuComplex **/, x /*const cuComplex **/,
// cublasCsyr2-NEXT:               incx /*int*/, y /*const cuComplex **/, incy /*int*/,
// cublasCsyr2-NEXT:               a /*cuComplex **/, lda /*int*/);
// cublasCsyr2-NEXT: Is migrated to:
// cublasCsyr2-NEXT:   oneapi::mkl::blas::column_major::syr2(*handle, upper_lower, n, dpct::get_value(alpha, *handle), (std::complex<float>*)x, incx, (std::complex<float>*)y, incy, (std::complex<float>*)a, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZsyrkx | FileCheck %s -check-prefix=cublasZsyrkx
// cublasZsyrkx: CUDA API:
// cublasZsyrkx-NEXT:   cublasZsyrkx(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZsyrkx-NEXT:                trans /*cublasOperation_t*/, n /*int*/, k /*int*/,
// cublasZsyrkx-NEXT:                alpha /*const cuDoubleComplex **/, a /*const cuDoubleComplex **/,
// cublasZsyrkx-NEXT:                lda /*int*/, b /*const cuDoubleComplex **/, ldb /*int*/,
// cublasZsyrkx-NEXT:                beta /*const cuDoubleComplex **/, c /*cuDoubleComplex **/,
// cublasZsyrkx-NEXT:                ldc /*int*/);
// cublasZsyrkx-NEXT: Is migrated to:
// cublasZsyrkx-NEXT:   dpct::syrk(*handle, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZsyr2 | FileCheck %s -check-prefix=cublasZsyr2
// cublasZsyr2: CUDA API:
// cublasZsyr2-NEXT:   cublasZsyr2(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZsyr2-NEXT:               n /*int*/, alpha /*const cuDoubleComplex **/,
// cublasZsyr2-NEXT:               x /*const cuDoubleComplex **/, incx /*int*/,
// cublasZsyr2-NEXT:               y /*const cuDoubleComplex **/, incy /*int*/,
// cublasZsyr2-NEXT:               a /*cuDoubleComplex **/, lda /*int*/);
// cublasZsyr2-NEXT: Is migrated to:
// cublasZsyr2-NEXT:   oneapi::mkl::blas::column_major::syr2(*handle, upper_lower, n, dpct::get_value(alpha, *handle), (std::complex<double>*)x, incx, (std::complex<double>*)y, incy, (std::complex<double>*)a, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCsscal | FileCheck %s -check-prefix=cublasCsscal
// cublasCsscal: CUDA API:
// cublasCsscal-NEXT:   cublasCsscal(handle /*cublasHandle_t*/, n /*int*/, alpha /*const float **/,
// cublasCsscal-NEXT:                x /*cuComplex **/, incx /*int*/);
// cublasCsscal-NEXT: Is migrated to:
// cublasCsscal-NEXT:   oneapi::mkl::blas::column_major::scal(*handle, n, dpct::get_value(alpha, *handle), (std::complex<float>*)x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasIdamin | FileCheck %s -check-prefix=cublasIdamin
// cublasIdamin: CUDA API:
// cublasIdamin-NEXT:   cublasIdamin(handle /*cublasHandle_t*/, n /*int*/, x /*const double **/,
// cublasIdamin-NEXT:                incx /*int*/, res /*int **/);
// cublasIdamin-NEXT: Is migrated to:
// cublasIdamin-NEXT:   int64_t* res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<int64_t>(1, dpct::get_default_queue());
// cublasIdamin-NEXT:   oneapi::mkl::blas::column_major::iamin(*handle, n, x, incx, res_temp_ptr_ct{{[0-9]+}}, oneapi::mkl::index_base::one).wait();
// cublasIdamin-NEXT:   int res_temp_host_ct{{[0-9]+}} = (int)*res_temp_ptr_ct{{[0-9]+}};
// cublasIdamin-NEXT:   dpct::dpct_memcpy(res, &res_temp_host_ct{{[0-9]+}}, sizeof(int));
// cublasIdamin-NEXT:   sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_default_queue());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSetMatrix | FileCheck %s -check-prefix=cublasSetMatrix
// cublasSetMatrix: CUDA API:
// cublasSetMatrix-NEXT:   cublasSetMatrix(rows /*int*/, cols /*int*/, elementsize /*int*/,
// cublasSetMatrix-NEXT:                   a /*const void **/, lda /*int*/, b /*void **/, ldb /*int*/);
// cublasSetMatrix-NEXT: Is migrated to:
// cublasSetMatrix-NEXT:   dpct::matrix_mem_copy((void*)b, (void*)a, ldb, lda, rows, cols, elementsize);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZscal | FileCheck %s -check-prefix=cublasZscal
// cublasZscal: CUDA API:
// cublasZscal-NEXT:   cublasZscal(handle /*cublasHandle_t*/, n /*int*/,
// cublasZscal-NEXT:               alpha /*const cuDoubleComplex **/, x /*cuDoubleComplex **/,
// cublasZscal-NEXT:               incx /*int*/);
// cublasZscal-NEXT: Is migrated to:
// cublasZscal-NEXT:   oneapi::mkl::blas::column_major::scal(*handle, n, dpct::get_value(alpha, *handle), (std::complex<double>*)x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCtrmv | FileCheck %s -check-prefix=cublasCtrmv
// cublasCtrmv: CUDA API:
// cublasCtrmv-NEXT:   cublasCtrmv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCtrmv-NEXT:               trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasCtrmv-NEXT:               n /*int*/, a /*const cuComplex **/, lda /*int*/,
// cublasCtrmv-NEXT:               x /*cuComplex **/, incx /*int*/);
// cublasCtrmv-NEXT: Is migrated to:
// cublasCtrmv-NEXT:   oneapi::mkl::blas::column_major::trmv(*handle, upper_lower, trans, unit_nonunit, n, (std::complex<float>*)a, lda, (std::complex<float>*)x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZrot | FileCheck %s -check-prefix=cublasZrot
// cublasZrot: CUDA API:
// cublasZrot-NEXT:   cublasZrot(handle /*cublasHandle_t*/, n /*int*/, x /*cuDoubleComplex **/,
// cublasZrot-NEXT:              incx /*int*/, y /*cuDoubleComplex **/, incy /*int*/,
// cublasZrot-NEXT:              c /*const double **/, s /*const cuDoubleComplex **/);
// cublasZrot-NEXT: Is migrated to:
// cublasZrot-NEXT:   dpct::rot(*handle, n, x, dpct::library_data_t::complex_double, incx, y, dpct::library_data_t::complex_double, incy, c, s, dpct::library_data_t::complex_double);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSetVector | FileCheck %s -check-prefix=cublasSetVector
// cublasSetVector: CUDA API:
// cublasSetVector-NEXT:   cublasSetVector(n /*int*/, elementsize /*int*/, x /*const void **/,
// cublasSetVector-NEXT:                   incx /*int*/, y /*void **/, incy /*int*/);
// cublasSetVector-NEXT: Is migrated to:
// cublasSetVector-NEXT:   dpct::matrix_mem_copy((void*)y, (void*)x, incy, incx, 1, n, elementsize);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSetStream | FileCheck %s -check-prefix=cublasSetStream
// cublasSetStream: CUDA API:
// cublasSetStream-NEXT:   cublasSetStream(handle /*cublasHandle_t*/, stream /*cudaStream_t*/);
// cublasSetStream-NEXT: Is migrated to:
// cublasSetStream-NEXT:   handle = stream;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZdgmm | FileCheck %s -check-prefix=cublasZdgmm
// cublasZdgmm: CUDA API:
// cublasZdgmm-NEXT:   cublasZdgmm(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasZdgmm-NEXT:               m /*int*/, n /*int*/, a /*const cuDoubleComplex **/, lda /*int*/,
// cublasZdgmm-NEXT:               x /*const cuDoubleComplex **/, incx /*int*/,
// cublasZdgmm-NEXT:               c /*cuDoubleComplex **/, ldc /*int*/);
// cublasZdgmm-NEXT: Is migrated to:
// cublasZdgmm-NEXT:   oneapi::mkl::blas::column_major::dgmm_batch(*handle, left_right, m, n, (std::complex<double>*)a, lda, 0, (std::complex<double>*)x, incx, 0, (std::complex<double>*)c, ldc, ldc * n, 1);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZdrot | FileCheck %s -check-prefix=cublasZdrot
// cublasZdrot: CUDA API:
// cublasZdrot-NEXT:   cublasZdrot(handle /*cublasHandle_t*/, n /*int*/, x /*cuDoubleComplex **/,
// cublasZdrot-NEXT:               incx /*int*/, y /*cuDoubleComplex **/, incy /*int*/,
// cublasZdrot-NEXT:               c /*const double **/, s /*const double **/);
// cublasZdrot-NEXT: Is migrated to:
// cublasZdrot-NEXT:   oneapi::mkl::blas::column_major::rot(*handle, n, (std::complex<double>*)x, incx, (std::complex<double>*)y, incy, dpct::get_value(c, *handle), dpct::get_value(s, *handle));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZhpr | FileCheck %s -check-prefix=cublasZhpr
// cublasZhpr: CUDA API:
// cublasZhpr-NEXT:   cublasZhpr(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZhpr-NEXT:              n /*int*/, alpha /*const double **/, x /*const cuDoubleComplex **/,
// cublasZhpr-NEXT:              incx /*int*/, a /*cuDoubleComplex **/);
// cublasZhpr-NEXT: Is migrated to:
// cublasZhpr-NEXT:   oneapi::mkl::blas::column_major::hpr(*handle, upper_lower, n, dpct::get_value(alpha, *handle), (std::complex<double>*)x, incx, (std::complex<double>*)a);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCtpmv | FileCheck %s -check-prefix=cublasCtpmv
// cublasCtpmv: CUDA API:
// cublasCtpmv-NEXT:   cublasCtpmv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCtpmv-NEXT:               trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasCtpmv-NEXT:               n /*int*/, a /*const cuComplex **/, x /*cuComplex **/,
// cublasCtpmv-NEXT:               incx /*int*/);
// cublasCtpmv-NEXT: Is migrated to:
// cublasCtpmv-NEXT:   oneapi::mkl::blas::column_major::tpmv(*handle, upper_lower, trans, unit_nonunit, n, (std::complex<float>*)a, (std::complex<float>*)x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDtrmv | FileCheck %s -check-prefix=cublasDtrmv
// cublasDtrmv: CUDA API:
// cublasDtrmv-NEXT:   cublasDtrmv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDtrmv-NEXT:               trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasDtrmv-NEXT:               n /*int*/, a /*const double **/, lda /*int*/, x /*double **/,
// cublasDtrmv-NEXT:               incx /*int*/);
// cublasDtrmv-NEXT: Is migrated to:
// cublasDtrmv-NEXT:   oneapi::mkl::blas::column_major::trmv(*handle, upper_lower, trans, unit_nonunit, n, a, lda, x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCtrsm | FileCheck %s -check-prefix=cublasCtrsm
// cublasCtrsm: CUDA API:
// cublasCtrsm-NEXT:   cublasCtrsm(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasCtrsm-NEXT:               upper_lower /*cublasFillMode_t*/, transa /*cublasOperation_t*/,
// cublasCtrsm-NEXT:               unit_diag /*cublasDiagType_t*/, m /*int*/, n /*int*/,
// cublasCtrsm-NEXT:               alpha /*const cuComplex **/, a /*const cuComplex **/, lda /*int*/,
// cublasCtrsm-NEXT:               b /*cuComplex **/, ldb /*int*/);
// cublasCtrsm-NEXT: Is migrated to:
// cublasCtrsm-NEXT:   oneapi::mkl::blas::column_major::trsm(*handle, left_right, upper_lower, transa, unit_diag, m, n, dpct::get_value(alpha, *handle), (std::complex<float>*)a, lda, (std::complex<float>*)b, ldb);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCsyrk | FileCheck %s -check-prefix=cublasCsyrk
// cublasCsyrk: CUDA API:
// cublasCsyrk-NEXT:   cublasCsyrk(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCsyrk-NEXT:               trans /*cublasOperation_t*/, n /*int*/, k /*int*/,
// cublasCsyrk-NEXT:               alpha /*const cuComplex **/, a /*const cuComplex **/, lda /*int*/,
// cublasCsyrk-NEXT:               beta /*const cuComplex **/, c /*cuComplex **/, ldc /*int*/);
// cublasCsyrk-NEXT: Is migrated to:
// cublasCsyrk-NEXT:   oneapi::mkl::blas::column_major::syrk(*handle, upper_lower, trans, n, k, dpct::get_value(alpha, *handle), (std::complex<float>*)a, lda, dpct::get_value(beta, *handle), (std::complex<float>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDsyr2 | FileCheck %s -check-prefix=cublasDsyr2
// cublasDsyr2: CUDA API:
// cublasDsyr2-NEXT:   cublasDsyr2(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDsyr2-NEXT:               n /*int*/, alpha /*const double **/, x /*const double **/,
// cublasDsyr2-NEXT:               incx /*int*/, y /*const double **/, incy /*int*/, a /*double **/,
// cublasDsyr2-NEXT:               lda /*int*/);
// cublasDsyr2-NEXT: Is migrated to:
// cublasDsyr2-NEXT:   oneapi::mkl::blas::column_major::syr2(*handle, upper_lower, n, dpct::get_value(alpha, *handle), x, incx, y, incy, a, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasIdamax | FileCheck %s -check-prefix=cublasIdamax
// cublasIdamax: CUDA API:
// cublasIdamax-NEXT:   cublasIdamax(handle /*cublasHandle_t*/, n /*int*/, x /*const double **/,
// cublasIdamax-NEXT:                incx /*int*/, res /*int **/);
// cublasIdamax-NEXT: Is migrated to:
// cublasIdamax-NEXT:   int64_t* res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<int64_t>(1, dpct::get_default_queue());
// cublasIdamax-NEXT:   oneapi::mkl::blas::column_major::iamax(*handle, n, x, incx, res_temp_ptr_ct{{[0-9]+}}, oneapi::mkl::index_base::one).wait();
// cublasIdamax-NEXT:   int res_temp_host_ct{{[0-9]+}} = (int)*res_temp_ptr_ct{{[0-9]+}};
// cublasIdamax-NEXT:   dpct::dpct_memcpy(res, &res_temp_host_ct{{[0-9]+}}, sizeof(int));
// cublasIdamax-NEXT:   sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_default_queue());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZgemv | FileCheck %s -check-prefix=cublasZgemv
// cublasZgemv: CUDA API:
// cublasZgemv-NEXT:   cublasZgemv(handle /*cublasHandle_t*/, trans /*cublasOperation_t*/, m /*int*/,
// cublasZgemv-NEXT:               n /*int*/, alpha /*const cuDoubleComplex **/,
// cublasZgemv-NEXT:               a /*const cuDoubleComplex **/, lda /*int*/,
// cublasZgemv-NEXT:               x /*const cuDoubleComplex **/, incx /*int*/,
// cublasZgemv-NEXT:               beta /*const cuDoubleComplex **/, y /*cuDoubleComplex **/,
// cublasZgemv-NEXT:               incy /*int*/);
// cublasZgemv-NEXT: Is migrated to:
// cublasZgemv-NEXT:   oneapi::mkl::blas::column_major::gemv(*handle, trans, m, n, dpct::get_value(alpha, *handle), (std::complex<double>*)a, lda, (std::complex<double>*)x, incx, dpct::get_value(beta, *handle), (std::complex<double>*)y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasIsamin | FileCheck %s -check-prefix=cublasIsamin
// cublasIsamin: CUDA API:
// cublasIsamin-NEXT:   cublasIsamin(handle /*cublasHandle_t*/, n /*int*/, x /*const float **/,
// cublasIsamin-NEXT:                incx /*int*/, res /*int **/);
// cublasIsamin-NEXT: Is migrated to:
// cublasIsamin-NEXT:   int64_t* res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<int64_t>(1, dpct::get_default_queue());
// cublasIsamin-NEXT:   oneapi::mkl::blas::column_major::iamin(*handle, n, x, incx, res_temp_ptr_ct{{[0-9]+}}, oneapi::mkl::index_base::one).wait();
// cublasIsamin-NEXT:   int res_temp_host_ct{{[0-9]+}} = (int)*res_temp_ptr_ct{{[0-9]+}};
// cublasIsamin-NEXT:   dpct::dpct_memcpy(res, &res_temp_host_ct{{[0-9]+}}, sizeof(int));
// cublasIsamin-NEXT:   sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_default_queue());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZswap | FileCheck %s -check-prefix=cublasZswap
// cublasZswap: CUDA API:
// cublasZswap-NEXT:   cublasZswap(handle /*cublasHandle_t*/, n /*int*/, x /*cuDoubleComplex **/,
// cublasZswap-NEXT:               incx /*int*/, y /*cuDoubleComplex **/, incy /*int*/);
// cublasZswap-NEXT: Is migrated to:
// cublasZswap-NEXT:   oneapi::mkl::blas::column_major::swap(*handle, n, (std::complex<double>*)x, incx, (std::complex<double>*)y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSspr | FileCheck %s -check-prefix=cublasSspr
// cublasSspr: CUDA API:
// cublasSspr-NEXT:   cublasSspr(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasSspr-NEXT:              n /*int*/, alpha /*const float **/, x /*const float **/,
// cublasSspr-NEXT:              incx /*int*/, a /*float **/);
// cublasSspr-NEXT: Is migrated to:
// cublasSspr-NEXT:   oneapi::mkl::blas::column_major::spr(*handle, upper_lower, n, dpct::get_value(alpha, *handle), x, incx, a);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDdot | FileCheck %s -check-prefix=cublasDdot
// cublasDdot: CUDA API:
// cublasDdot-NEXT:   cublasDdot(handle /*cublasHandle_t*/, n /*int*/, x /*const double **/,
// cublasDdot-NEXT:              incx /*int*/, y /*const double **/, incy /*int*/,
// cublasDdot-NEXT:              res /*double **/);
// cublasDdot-NEXT: Is migrated to:
// cublasDdot-NEXT:   double* res_temp_ptr_ct{{[0-9]+}} = res;
// cublasDdot-NEXT:   if(sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasDdot-NEXT:     res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<double>(1, dpct::get_default_queue());
// cublasDdot-NEXT:   }
// cublasDdot-NEXT:   oneapi::mkl::blas::column_major::dot(*handle, n, x, incx, y, incy, res_temp_ptr_ct{{[0-9]+}});
// cublasDdot-NEXT:   if(sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasDdot-NEXT:     handle->wait();
// cublasDdot-NEXT:     *res = *res_temp_ptr_ct{{[0-9]+}};
// cublasDdot-NEXT:     sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_default_queue());
// cublasDdot-NEXT:   }

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCsymv | FileCheck %s -check-prefix=cublasCsymv
// cublasCsymv: CUDA API:
// cublasCsymv-NEXT:   cublasCsymv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCsymv-NEXT:               n /*int*/, alpha /*const cuComplex **/, a /*const cuComplex **/,
// cublasCsymv-NEXT:               lda /*int*/, x /*const cuComplex **/, incx /*int*/,
// cublasCsymv-NEXT:               beta /*const cuComplex **/, y /*cuComplex **/, incy /*int*/);
// cublasCsymv-NEXT: Is migrated to:
// cublasCsymv-NEXT:   oneapi::mkl::blas::column_major::symv(*handle, upper_lower, n, dpct::get_value(alpha, *handle), (std::complex<float>*)a, lda, (std::complex<float>*)x, incx, dpct::get_value(beta, *handle), (std::complex<float>*)y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDspr2 | FileCheck %s -check-prefix=cublasDspr2
// cublasDspr2: CUDA API:
// cublasDspr2-NEXT:   cublasDspr2(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDspr2-NEXT:               n /*int*/, alpha /*const double **/, x /*const double **/,
// cublasDspr2-NEXT:               incx /*int*/, y /*const double **/, incy /*int*/, a /*double **/);
// cublasDspr2-NEXT: Is migrated to:
// cublasDspr2-NEXT:   oneapi::mkl::blas::column_major::spr2(*handle, upper_lower, n, dpct::get_value(alpha, *handle), x, incx, y, incy, a);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZherk | FileCheck %s -check-prefix=cublasZherk
// cublasZherk: CUDA API:
// cublasZherk-NEXT:   cublasZherk(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZherk-NEXT:               trans /*cublasOperation_t*/, n /*int*/, k /*int*/,
// cublasZherk-NEXT:               alpha /*const double **/, a /*const cuDoubleComplex **/,
// cublasZherk-NEXT:               lda /*int*/, beta /*const double **/, c /*cuDoubleComplex **/,
// cublasZherk-NEXT:               ldc /*int*/);
// cublasZherk-NEXT: Is migrated to:
// cublasZherk-NEXT:   oneapi::mkl::blas::column_major::herk(*handle, upper_lower, trans, n, k, dpct::get_value(alpha, *handle), (std::complex<double>*)a, lda, dpct::get_value(beta, *handle), (std::complex<double>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZher2k | FileCheck %s -check-prefix=cublasZher2k
// cublasZher2k: CUDA API:
// cublasZher2k-NEXT:   cublasZher2k(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasZher2k-NEXT:                trans /*cublasOperation_t*/, n /*int*/, k /*int*/,
// cublasZher2k-NEXT:                alpha /*const cuDoubleComplex **/, a /*const cuDoubleComplex **/,
// cublasZher2k-NEXT:                lda /*int*/, b /*const cuDoubleComplex **/, ldb /*int*/,
// cublasZher2k-NEXT:                beta /*const double **/, c /*cuDoubleComplex **/, ldc /*int*/);
// cublasZher2k-NEXT: Is migrated to:
// cublasZher2k-NEXT:   oneapi::mkl::blas::column_major::her2k(*handle, upper_lower, trans, n, k, dpct::get_value(alpha, *handle), (std::complex<double>*)a, lda, (std::complex<double>*)b, ldb, dpct::get_value(beta, *handle), (std::complex<double>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDsyr2k | FileCheck %s -check-prefix=cublasDsyr2k
// cublasDsyr2k: CUDA API:
// cublasDsyr2k-NEXT:   cublasDsyr2k(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasDsyr2k-NEXT:                trans /*cublasOperation_t*/, n /*int*/, k /*int*/,
// cublasDsyr2k-NEXT:                alpha /*const double **/, a /*const double **/, lda /*int*/,
// cublasDsyr2k-NEXT:                b /*const double **/, ldb /*int*/, beta /*const double **/,
// cublasDsyr2k-NEXT:                c /*double **/, ldc /*int*/);
// cublasDsyr2k-NEXT: Is migrated to:
// cublasDsyr2k-NEXT:   oneapi::mkl::blas::column_major::syr2k(*handle, upper_lower, trans, n, k, dpct::get_value(alpha, *handle), a, lda, b, ldb, dpct::get_value(beta, *handle), c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDgemm | FileCheck %s -check-prefix=cublasDgemm
// cublasDgemm: CUDA API:
// cublasDgemm-NEXT:   cublasDgemm(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasDgemm-NEXT:               transb /*cublasOperation_t*/, m /*int*/, n /*int*/, k /*int*/,
// cublasDgemm-NEXT:               alpha /*const double **/, a /*const double **/, lda /*int*/,
// cublasDgemm-NEXT:               b /*const double **/, ldb /*int*/, beta /*const double **/,
// cublasDgemm-NEXT:               c /*double **/, ldc /*int*/);
// cublasDgemm-NEXT: Is migrated to:
// cublasDgemm-NEXT:   oneapi::mkl::blas::column_major::gemm(*handle, transa, transb, m, n, k, dpct::get_value(alpha, *handle), a, lda, b, ldb, dpct::get_value(beta, *handle), c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasGetPointerMode | FileCheck %s -check-prefix=cublasGetPointerMode
// cublasGetPointerMode: CUDA API:
// cublasGetPointerMode-NEXT:   cublasGetPointerMode(handle /*cublasHandle_t*/,
// cublasGetPointerMode-NEXT:                        host_device /*cublasPointerMode_t **/);
// cublasGetPointerMode-NEXT: Is migrated to:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSrotg | FileCheck %s -check-prefix=cublasSrotg
// cublasSrotg: CUDA API:
// cublasSrotg-NEXT:   cublasSrotg(handle /*cublasHandle_t*/, a /*float **/, b /*float **/,
// cublasSrotg-NEXT:               c /*float **/, s /*float **/);
// cublasSrotg-NEXT: Is migrated to:
// cublasSrotg-NEXT:   float* a_ct{{[0-9]+}} = a;
// cublasSrotg-NEXT:   float* b_ct{{[0-9]+}} = b;
// cublasSrotg-NEXT:   float* c_ct{{[0-9]+}} = c;
// cublasSrotg-NEXT:   float* s_ct{{[0-9]+}} = s;
// cublasSrotg-NEXT:   if(sycl::get_pointer_type(a, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(a, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasSrotg-NEXT:     a_ct{{[0-9]+}} = sycl::malloc_shared<float>(4, dpct::get_default_queue());
// cublasSrotg-NEXT:     b_ct{{[0-9]+}} = a_ct{{[0-9]+}} + 1;
// cublasSrotg-NEXT:     c_ct{{[0-9]+}} = a_ct{{[0-9]+}} + 2;
// cublasSrotg-NEXT:     s_ct{{[0-9]+}} = a_ct{{[0-9]+}} + 3;
// cublasSrotg-NEXT:     *a_ct{{[0-9]+}} = *a;
// cublasSrotg-NEXT:     *b_ct{{[0-9]+}} = *b;
// cublasSrotg-NEXT:     *c_ct{{[0-9]+}} = *c;
// cublasSrotg-NEXT:     *s_ct{{[0-9]+}} = *s;
// cublasSrotg-NEXT:   }
// cublasSrotg-NEXT:   oneapi::mkl::blas::column_major::rotg(*handle, a_ct{{[0-9]+}}, b_ct{{[0-9]+}}, c_ct{{[0-9]+}}, s_ct{{[0-9]+}});
// cublasSrotg-NEXT:   if(sycl::get_pointer_type(a, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(a, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasSrotg-NEXT:     handle->wait();
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
// cublasSgemmEx-NEXT:   dpct::gemm(*handle, transa, transb, m, n, k, alpha, a, atype, lda, b, btype, ldb, beta, c, ctype, ldc, dpct::library_data_t::real_float);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDrotmg | FileCheck %s -check-prefix=cublasDrotmg
// cublasDrotmg: CUDA API:
// cublasDrotmg-NEXT:   cublasDrotmg(handle /*cublasHandle_t*/, d1 /*double **/, d2 /*double **/,
// cublasDrotmg-NEXT:                x1 /*double **/, y1 /*const double **/, param /*double **/);
// cublasDrotmg-NEXT: Is migrated to:
// cublasDrotmg-NEXT:   double* d1_ct{{[0-9]+}} = d1;
// cublasDrotmg-NEXT:   double* d2_ct{{[0-9]+}} = d2;
// cublasDrotmg-NEXT:   double* x1_ct{{[0-9]+}} = x1;
// cublasDrotmg-NEXT:   double* param_ct{{[0-9]+}} = param;
// cublasDrotmg-NEXT:   if(sycl::get_pointer_type(d1, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(d1, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasDrotmg-NEXT:     d1_ct{{[0-9]+}} = sycl::malloc_shared<double>(8, dpct::get_default_queue());
// cublasDrotmg-NEXT:     d2_ct{{[0-9]+}} = d1_ct{{[0-9]+}} + 1;
// cublasDrotmg-NEXT:     x1_ct{{[0-9]+}} = d1_ct{{[0-9]+}} + 2;
// cublasDrotmg-NEXT:     param_ct{{[0-9]+}} = d1_ct{{[0-9]+}} + 3;
// cublasDrotmg-NEXT:     *d1_ct{{[0-9]+}} = *d1;
// cublasDrotmg-NEXT:     *d2_ct{{[0-9]+}} = *d2;
// cublasDrotmg-NEXT:     *x1_ct{{[0-9]+}} = *x1;
// cublasDrotmg-NEXT:   }
// cublasDrotmg-NEXT:   oneapi::mkl::blas::column_major::rotmg(*handle, d1_ct{{[0-9]+}}, d2_ct{{[0-9]+}}, x1_ct{{[0-9]+}}, dpct::get_value(y1, *handle), param_ct{{[0-9]+}});
// cublasDrotmg-NEXT:   if(sycl::get_pointer_type(d1, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(d1, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasDrotmg-NEXT:     handle->wait();
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
// cublasStpmv-NEXT:   oneapi::mkl::blas::column_major::tpmv(*handle, upper_lower, trans, unit_nonunit, n, a, x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasStrsv | FileCheck %s -check-prefix=cublasStrsv
// cublasStrsv: CUDA API:
// cublasStrsv-NEXT:   cublasStrsv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasStrsv-NEXT:               trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasStrsv-NEXT:               n /*int*/, a /*const float **/, lda /*int*/, x /*float **/,
// cublasStrsv-NEXT:               incx /*int*/);
// cublasStrsv-NEXT: Is migrated to:
// cublasStrsv-NEXT:   oneapi::mkl::blas::column_major::trsv(*handle, upper_lower, trans, unit_nonunit, n, a, lda, x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasStbmv | FileCheck %s -check-prefix=cublasStbmv
// cublasStbmv: CUDA API:
// cublasStbmv-NEXT:   cublasStbmv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasStbmv-NEXT:               trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasStbmv-NEXT:               n /*int*/, k /*int*/, a /*const float **/, lda /*int*/,
// cublasStbmv-NEXT:               x /*float **/, incx /*int*/);
// cublasStbmv-NEXT: Is migrated to:
// cublasStbmv-NEXT:   oneapi::mkl::blas::column_major::tbmv(*handle, upper_lower, trans, unit_nonunit, n, k, a, lda, x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasStpsv | FileCheck %s -check-prefix=cublasStpsv
// cublasStpsv: CUDA API:
// cublasStpsv-NEXT:   cublasStpsv(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasStpsv-NEXT:               trans /*cublasOperation_t*/, unit_nonunit /*cublasDiagType_t*/,
// cublasStpsv-NEXT:               n /*int*/, a /*const float **/, x /*float **/, incx /*int*/);
// cublasStpsv-NEXT: Is migrated to:
// cublasStpsv-NEXT:   oneapi::mkl::blas::column_major::tpsv(*handle, upper_lower, trans, unit_nonunit, n, a, x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCherkx | FileCheck %s -check-prefix=cublasCherkx
// cublasCherkx: CUDA API:
// cublasCherkx-NEXT:   cublasCherkx(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCherkx-NEXT:                trans /*cublasOperation_t*/, n /*int*/, k /*int*/,
// cublasCherkx-NEXT:                alpha /*const cuComplex **/, a /*const cuComplex **/,
// cublasCherkx-NEXT:                lda /*int*/, b /*const cuComplex **/, ldb /*int*/,
// cublasCherkx-NEXT:                beta /*const float **/, c /*cuComplex **/, ldc /*int*/);
// cublasCherkx-NEXT: Is migrated to:
// cublasCherkx-NEXT:   dpct::herk(*handle, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCgeam | FileCheck %s -check-prefix=cublasCgeam
// cublasCgeam: CUDA API:
// cublasCgeam-NEXT:   cublasCgeam(handle /*cublasHandle_t*/, transa /*cublasOperation_t*/,
// cublasCgeam-NEXT:               transb /*cublasOperation_t*/, m /*int*/, n /*int*/,
// cublasCgeam-NEXT:               alpha /*const cuComplex **/, a /*const cuComplex **/, lda /*int*/,
// cublasCgeam-NEXT:               beta /*const cuComplex **/, b /*const cuComplex **/, ldb /*int*/,
// cublasCgeam-NEXT:               c /*cuComplex **/, ldc /*int*/);
// cublasCgeam-NEXT: Is migrated to:
// cublasCgeam-NEXT:   oneapi::mkl::blas::column_major::omatadd(*handle, transa, transb, m, n, dpct::get_value(alpha, *handle), (std::complex<float>*)a, lda, dpct::get_value(beta, *handle), (std::complex<float>*)b, ldb, (std::complex<float>*)c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasZrotg | FileCheck %s -check-prefix=cublasZrotg
// cublasZrotg: CUDA API:
// cublasZrotg-NEXT:   cublasZrotg(handle /*cublasHandle_t*/, a /*cuDoubleComplex **/,
// cublasZrotg-NEXT:               b /*cuDoubleComplex **/, c /*double **/, s /*cuDoubleComplex **/);
// cublasZrotg-NEXT: Is migrated to:
// cublasZrotg-NEXT:   sycl::double2* a_ct{{[0-9]+}} = a;
// cublasZrotg-NEXT:   sycl::double2* b_ct{{[0-9]+}} = b;
// cublasZrotg-NEXT:   double* c_ct{{[0-9]+}} = c;
// cublasZrotg-NEXT:   sycl::double2* s_ct{{[0-9]+}} = s;
// cublasZrotg-NEXT:   if(sycl::get_pointer_type(a, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(a, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasZrotg-NEXT:     a_ct{{[0-9]+}} = sycl::malloc_shared<sycl::double2>(3, dpct::get_default_queue());
// cublasZrotg-NEXT:     c_ct{{[0-9]+}} = sycl::malloc_shared<double>(1, dpct::get_default_queue());
// cublasZrotg-NEXT:     b_ct{{[0-9]+}} = a_ct{{[0-9]+}} + 1;
// cublasZrotg-NEXT:     s_ct{{[0-9]+}} = a_ct{{[0-9]+}} + 2;
// cublasZrotg-NEXT:     *a_ct{{[0-9]+}} = *a;
// cublasZrotg-NEXT:     *b_ct{{[0-9]+}} = *b;
// cublasZrotg-NEXT:     *c_ct{{[0-9]+}} = *c;
// cublasZrotg-NEXT:     *s_ct{{[0-9]+}} = *s;
// cublasZrotg-NEXT:   }
// cublasZrotg-NEXT:   oneapi::mkl::blas::column_major::rotg(*handle, (std::complex<double>*)a_ct{{[0-9]+}}, (std::complex<double>*)b_ct{{[0-9]+}}, c_ct{{[0-9]+}}, (std::complex<double>*)s_ct{{[0-9]+}});
// cublasZrotg-NEXT:   if(sycl::get_pointer_type(a, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(a, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasZrotg-NEXT:     handle->wait();
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
// cublasHgemm-NEXT:   oneapi::mkl::blas::column_major::gemm(*handle, transa, transb, m, n, k, dpct::get_value(alpha, *handle), a, lda, b, ldb, dpct::get_value(beta, *handle), c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSgemv | FileCheck %s -check-prefix=cublasSgemv
// cublasSgemv: CUDA API:
// cublasSgemv-NEXT:   cublasSgemv(handle /*cublasHandle_t*/, trans /*cublasOperation_t*/, m /*int*/,
// cublasSgemv-NEXT:               n /*int*/, alpha /*const float **/, a /*const float **/,
// cublasSgemv-NEXT:               lda /*int*/, x /*const float **/, incx /*int*/,
// cublasSgemv-NEXT:               beta /*const float **/, y /*float **/, incy /*int*/);
// cublasSgemv-NEXT: Is migrated to:
// cublasSgemv-NEXT:   oneapi::mkl::blas::column_major::gemv(*handle, trans, m, n, dpct::get_value(alpha, *handle), a, lda, x, incx, dpct::get_value(beta, *handle), y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasSrotmg | FileCheck %s -check-prefix=cublasSrotmg
// cublasSrotmg: CUDA API:
// cublasSrotmg-NEXT:   cublasSrotmg(handle /*cublasHandle_t*/, d1 /*float **/, d2 /*float **/,
// cublasSrotmg-NEXT:                x1 /*float **/, y1 /*const float **/, param /*float **/);
// cublasSrotmg-NEXT: Is migrated to:
// cublasSrotmg-NEXT:   float* d1_ct{{[0-9]+}} = d1;
// cublasSrotmg-NEXT:   float* d2_ct{{[0-9]+}} = d2;
// cublasSrotmg-NEXT:   float* x1_ct{{[0-9]+}} = x1;
// cublasSrotmg-NEXT:   float* param_ct{{[0-9]+}} = param;
// cublasSrotmg-NEXT:   if(sycl::get_pointer_type(d1, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(d1, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasSrotmg-NEXT:     d1_ct{{[0-9]+}} = sycl::malloc_shared<float>(8, dpct::get_default_queue());
// cublasSrotmg-NEXT:     d2_ct{{[0-9]+}} = d1_ct{{[0-9]+}} + 1;
// cublasSrotmg-NEXT:     x1_ct{{[0-9]+}} = d1_ct{{[0-9]+}} + 2;
// cublasSrotmg-NEXT:     param_ct{{[0-9]+}} = d1_ct{{[0-9]+}} + 3;
// cublasSrotmg-NEXT:     *d1_ct{{[0-9]+}} = *d1;
// cublasSrotmg-NEXT:     *d2_ct{{[0-9]+}} = *d2;
// cublasSrotmg-NEXT:     *x1_ct{{[0-9]+}} = *x1;
// cublasSrotmg-NEXT:   }
// cublasSrotmg-NEXT:   oneapi::mkl::blas::column_major::rotmg(*handle, d1_ct{{[0-9]+}}, d2_ct{{[0-9]+}}, x1_ct{{[0-9]+}}, dpct::get_value(y1, *handle), param_ct{{[0-9]+}});
// cublasSrotmg-NEXT:   if(sycl::get_pointer_type(d1, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(d1, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasSrotmg-NEXT:     handle->wait();
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
// cublasZhbmv-NEXT:   oneapi::mkl::blas::column_major::hbmv(*handle, upper_lower, n, k, dpct::get_value(alpha, *handle), (std::complex<double>*)a, lda, (std::complex<double>*)x, incx, dpct::get_value(beta, *handle), (std::complex<double>*)y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCswap | FileCheck %s -check-prefix=cublasCswap
// cublasCswap: CUDA API:
// cublasCswap-NEXT:   cublasCswap(handle /*cublasHandle_t*/, n /*int*/, x /*cuComplex **/,
// cublasCswap-NEXT:               incx /*int*/, y /*cuComplex **/, incy /*int*/);
// cublasCswap-NEXT: Is migrated to:
// cublasCswap-NEXT:   oneapi::mkl::blas::column_major::swap(*handle, n, (std::complex<float>*)x, incx, (std::complex<float>*)y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCher2 | FileCheck %s -check-prefix=cublasCher2
// cublasCher2: CUDA API:
// cublasCher2-NEXT:   cublasCher2(handle /*cublasHandle_t*/, upper_lower /*cublasFillMode_t*/,
// cublasCher2-NEXT:               n /*int*/, alpha /*const cuComplex **/, x /*const cuComplex **/,
// cublasCher2-NEXT:               incx /*int*/, y /*const cuComplex **/, incy /*int*/,
// cublasCher2-NEXT:               a /*cuComplex **/, lda /*int*/);
// cublasCher2-NEXT: Is migrated to:
// cublasCher2-NEXT:   oneapi::mkl::blas::column_major::her2(*handle, upper_lower, n, dpct::get_value(alpha, *handle), (std::complex<float>*)x, incx, (std::complex<float>*)y, incy, (std::complex<float>*)a, lda);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCaxpy | FileCheck %s -check-prefix=cublasCaxpy
// cublasCaxpy: CUDA API:
// cublasCaxpy-NEXT:   cublasCaxpy(handle /*cublasHandle_t*/, n /*int*/, alpha /*const cuComplex **/,
// cublasCaxpy-NEXT:               x /*const cuComplex **/, incx /*int*/, y /*cuComplex **/,
// cublasCaxpy-NEXT:               incy /*int*/);
// cublasCaxpy-NEXT: Is migrated to:
// cublasCaxpy-NEXT:   oneapi::mkl::blas::column_major::axpy(*handle, n, dpct::get_value(alpha, *handle), (std::complex<float>*)x, incx, (std::complex<float>*)y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDaxpy | FileCheck %s -check-prefix=cublasDaxpy
// cublasDaxpy: CUDA API:
// cublasDaxpy-NEXT:   cublasDaxpy(handle /*cublasHandle_t*/, n /*int*/, alpha /*const double **/,
// cublasDaxpy-NEXT:               x /*const double **/, incx /*int*/, y /*double **/, incy /*int*/);
// cublasDaxpy-NEXT: Is migrated to:
// cublasDaxpy-NEXT:   oneapi::mkl::blas::column_major::axpy(*handle, n, dpct::get_value(alpha, *handle), x, incx, y, incy);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasDscal | FileCheck %s -check-prefix=cublasDscal
// cublasDscal: CUDA API:
// cublasDscal-NEXT:   cublasDscal(handle /*cublasHandle_t*/, n /*int*/, alpha /*const double **/,
// cublasDscal-NEXT:               x /*double **/, incx /*int*/);
// cublasDscal-NEXT: Is migrated to:
// cublasDscal-NEXT:   oneapi::mkl::blas::column_major::scal(*handle, n, dpct::get_value(alpha, *handle), x, incx);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCtrmm | FileCheck %s -check-prefix=cublasCtrmm
// cublasCtrmm: CUDA API:
// cublasCtrmm-NEXT:   cublasCtrmm(handle /*cublasHandle_t*/, left_right /*cublasSideMode_t*/,
// cublasCtrmm-NEXT:               upper_lower /*cublasFillMode_t*/, transa /*cublasOperation_t*/,
// cublasCtrmm-NEXT:               unit_diag /*cublasDiagType_t*/, m /*int*/, n /*int*/,
// cublasCtrmm-NEXT:               alpha /*const cuComplex **/, a /*const cuComplex **/, lda /*int*/,
// cublasCtrmm-NEXT:               b /*const cuComplex **/, ldb /*int*/, c /*cuComplex **/,
// cublasCtrmm-NEXT:               ldc /*int*/);
// cublasCtrmm-NEXT: Is migrated to:
// cublasCtrmm-NEXT:   dpct::trmm(*handle, left_right, upper_lower, transa, unit_diag, m, n, alpha, a, lda, b, ldb, c, ldc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasCgbmv | FileCheck %s -check-prefix=cublasCgbmv
// cublasCgbmv: CUDA API:
// cublasCgbmv-NEXT:   cublasCgbmv(handle /*cublasHandle_t*/, trans /*cublasOperation_t*/, m /*int*/,
// cublasCgbmv-NEXT:               n /*int*/, kl /*int*/, ku /*int*/, alpha /*const cuComplex **/,
// cublasCgbmv-NEXT:               a /*const cuComplex **/, lda /*int*/, x /*const cuComplex **/,
// cublasCgbmv-NEXT:               incx /*int*/, beta /*const cuComplex **/, y /*cuComplex **/,
// cublasCgbmv-NEXT:               incy /*int*/);
// cublasCgbmv-NEXT: Is migrated to:
// cublasCgbmv-NEXT:   oneapi::mkl::blas::column_major::gbmv(*handle, trans, m, n, kl, ku, dpct::get_value(alpha, *handle), (std::complex<float>*)a, lda, (std::complex<float>*)x, incx, dpct::get_value(beta, *handle), (std::complex<float>*)y, incy);

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
// cublasScasum-NEXT:     res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<float>(1, dpct::get_default_queue());
// cublasScasum-NEXT:   }
// cublasScasum-NEXT:   oneapi::mkl::blas::column_major::asum(*handle, n, (std::complex<float>*)x, incx, res_temp_ptr_ct{{[0-9]+}});
// cublasScasum-NEXT:   if(sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasScasum-NEXT:     handle->wait();
// cublasScasum-NEXT:     *res = *res_temp_ptr_ct{{[0-9]+}};
// cublasScasum-NEXT:     sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_default_queue());
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
// cublasZdotu-NEXT:     res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<sycl::double2>(1, dpct::get_default_queue());
// cublasZdotu-NEXT:   }
// cublasZdotu-NEXT:   oneapi::mkl::blas::column_major::dotu(*handle, n, (std::complex<double>*)x, incx, (std::complex<double>*)y, incy, (std::complex<double>*)res_temp_ptr_ct{{[0-9]+}});
// cublasZdotu-NEXT:   if(sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasZdotu-NEXT:     handle->wait();
// cublasZdotu-NEXT:     *res = *res_temp_ptr_ct{{[0-9]+}};
// cublasZdotu-NEXT:     sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_default_queue());
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
// cublasZdotc-NEXT:     res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<sycl::double2>(1, dpct::get_default_queue());
// cublasZdotc-NEXT:   }
// cublasZdotc-NEXT:   oneapi::mkl::blas::column_major::dotc(*handle, n, (std::complex<double>*)x, incx, (std::complex<double>*)y, incy, (std::complex<double>*)res_temp_ptr_ct{{[0-9]+}});
// cublasZdotc-NEXT:   if(sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::device && sycl::get_pointer_type(res, handle->get_context())!=sycl::usm::alloc::shared) {
// cublasZdotc-NEXT:     handle->wait();
// cublasZdotc-NEXT:     *res = *res_temp_ptr_ct{{[0-9]+}};
// cublasZdotc-NEXT:     sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_default_queue());
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
// cublasSetPointerMode-NEXT: Is migrated to:

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

