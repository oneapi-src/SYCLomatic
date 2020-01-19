// RUN: dpct --format-range=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cublasTtrmm.dp.cpp --match-full-lines %s
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

int main(){
  cublasStatus_t status;
  cublasHandle_t handle;
  int n = 275;
  int m = 275;
  int lda = 275;
  int ldb = 275;
  int ldc = 275;
  const float *A_S = 0;
  const float *B_S = 0;
  float *C_S = 0;
  float alpha_S = 1.0f;
  const double *A_D = 0;
  const double *B_D = 0;
  double *C_D = 0;
  double alpha_D = 1.0;

  int side0 = 0; int side1 = 1; int fill0 = 0; int fill1 = 1;
  int trans0 = 0; int trans1 = 1; int trans2 = 2; int diag0 = 0; int diag1 = 1;
  // CHECK: /*
  // CHECK-NEXT: DPCT1003:0: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto transpose_ct3 = trans0;
  // CHECK-NEXT: auto ptr_ct8 = A_S;
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(ptr_ct8);
  // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto ptr_ct12 = C_S;
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(ptr_ct12);
  // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto ld_ct13 = ldc; auto m_ct5 = m; auto n_ct6 = n;
  // CHECK-NEXT: dpct::matrix_mem_copy(ptr_ct12, B_S, ld_ct13, ldb, m_ct5, n_ct6, dpct::device_to_device, handle);
  // CHECK-NEXT: status = (mkl::blas::trmm(handle, (mkl::side)side0, (((int)fill0)==0?(mkl::uplo::lower):(mkl::uplo::upper)), (((int)transpose_ct3)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct3)), (mkl::diag)diag0, m_ct5, n_ct6, *(&alpha_S), buffer_ct{{[0-9]+}}, lda,  buffer_ct{{[0-9]+}}, ld_ct13), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto ptr_ct8 = A_S;
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(ptr_ct8);
  // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto ptr_ct12 = C_S;
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(ptr_ct12);
  // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(float)));
  // CHECK-NEXT: auto ld_ct13 = ldc; auto m_ct5 = m; auto n_ct6 = n;
  // CHECK-NEXT: dpct::matrix_mem_copy(ptr_ct12, B_S, ld_ct13, ldb, m_ct5, n_ct6, dpct::device_to_device, handle);
  // CHECK-NEXT: mkl::blas::trmm(handle, mkl::side::left, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::nonunit, m_ct5, n_ct6, *(&alpha_S), buffer_ct{{[0-9]+}}, lda,  buffer_ct{{[0-9]+}}, ld_ct13);
  // CHECK-NEXT: }
  status = cublasStrmm(handle, (cublasSideMode_t)side0, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, m, n, &alpha_S, A_S, lda, B_S, ldb, C_S, ldc);
  cublasStrmm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &alpha_S, A_S, lda, B_S, ldb, C_S, ldc);


  // CHECK: /*
  // CHECK-NEXT: DPCT1003:1: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto transpose_ct3 = trans1;
  // CHECK-NEXT: auto ptr_ct8 = A_D;
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(ptr_ct8);
  // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto ptr_ct12 = C_D;
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(ptr_ct12);
  // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto ld_ct13 = ldc; auto m_ct5 = m; auto n_ct6 = n;
  // CHECK-NEXT: dpct::matrix_mem_copy(ptr_ct12, B_D, ld_ct13, ldb, m_ct5, n_ct6, dpct::device_to_device, handle);
  // CHECK-NEXT: status = (mkl::blas::trmm(handle, (mkl::side)side1, (((int)fill1)==0?(mkl::uplo::lower):(mkl::uplo::upper)), (((int)transpose_ct3)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct3)), (mkl::diag)diag1, m_ct5, n_ct6, *(&alpha_D), buffer_ct{{[0-9]+}}, lda,  buffer_ct{{[0-9]+}}, ld_ct13), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto ptr_ct8 = A_D;
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(ptr_ct8);
  // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto ptr_ct12 = C_D;
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(ptr_ct12);
  // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT: auto ld_ct13 = ldc; auto m_ct5 = m; auto n_ct6 = n;
  // CHECK-NEXT: dpct::matrix_mem_copy(ptr_ct12, B_D, ld_ct13, ldb, m_ct5, n_ct6, dpct::device_to_device, handle);
  // CHECK-NEXT: mkl::blas::trmm(handle, mkl::side::left, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::nonunit, m_ct5, n_ct6, *(&alpha_D), buffer_ct{{[0-9]+}}, lda,  buffer_ct{{[0-9]+}}, ld_ct13);
  // CHECK-NEXT: }
  status = cublasDtrmm(handle, (cublasSideMode_t)side1, (cublasFillMode_t)fill1, (cublasOperation_t)trans1, (cublasDiagType_t)diag1, m, n, &alpha_D, A_D, lda, B_D, ldb, C_D, ldc);
  cublasDtrmm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &alpha_D, A_D, lda, B_D, ldb, C_D, ldc);


  const cuComplex *A_C = 0;
  const cuComplex *B_C = 0;
  cuComplex *C_C = 0;
  cuComplex alpha_C = make_cuComplex(1.0f,0.0f);
  const cuDoubleComplex *A_Z = 0;
  const cuDoubleComplex *B_Z = 0;
  cuDoubleComplex *C_Z = 0;
  cuDoubleComplex alpha_Z = make_cuDoubleComplex(1.0,0.0);


  // CHECK: /*
  // CHECK-NEXT: DPCT1003:2: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto transpose_ct3 = trans2;
  // CHECK-NEXT: auto ptr_ct8 = A_C;
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(ptr_ct8);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto ptr_ct12 = C_C;
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(ptr_ct12);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto ld_ct13 = ldc; auto m_ct5 = m; auto n_ct6 = n;
  // CHECK-NEXT: dpct::matrix_mem_copy(ptr_ct12, B_C, ld_ct13, ldb, m_ct5, n_ct6, dpct::device_to_device, handle);
  // CHECK-NEXT: status = (mkl::blas::trmm(handle, (mkl::side)0, (((int)0)==0?(mkl::uplo::lower):(mkl::uplo::upper)), (((int)transpose_ct3)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct3)), (mkl::diag)0, m_ct5, n_ct6, std::complex<float>((&alpha_C)->x(),(&alpha_C)->y()), buffer_ct{{[0-9]+}}, lda,  buffer_ct{{[0-9]+}}, ld_ct13), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto ptr_ct8 = A_C;
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(ptr_ct8);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto ptr_ct12 = C_C;
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(ptr_ct12);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT: auto ld_ct13 = ldc; auto m_ct5 = m; auto n_ct6 = n;
  // CHECK-NEXT: dpct::matrix_mem_copy(ptr_ct12, B_C, ld_ct13, ldb, m_ct5, n_ct6, dpct::device_to_device, handle);
  // CHECK-NEXT: mkl::blas::trmm(handle, mkl::side::left, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::nonunit, m_ct5, n_ct6, std::complex<float>((&alpha_C)->x(),(&alpha_C)->y()), buffer_ct{{[0-9]+}}, lda,  buffer_ct{{[0-9]+}}, ld_ct13);
  // CHECK-NEXT: }
  status = cublasCtrmm(handle, (cublasSideMode_t)0, (cublasFillMode_t)0, (cublasOperation_t)trans2, (cublasDiagType_t)0, m, n, &alpha_C, A_C, lda, B_C, ldb, C_C, ldc);
  cublasCtrmm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &alpha_C, A_C, lda, B_C, ldb, C_C, ldc);


  // CHECK: /*
  // CHECK-NEXT: DPCT1003:3: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: {
  // CHECK-NEXT: auto transpose_ct3 = 2;
  // CHECK-NEXT: auto ptr_ct8 = A_Z;
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(ptr_ct8);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto ptr_ct12 = C_Z;
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(ptr_ct12);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto ld_ct13 = ldc; auto m_ct5 = m; auto n_ct6 = n;
  // CHECK-NEXT: dpct::matrix_mem_copy(ptr_ct12, B_Z, ld_ct13, ldb, m_ct5, n_ct6, dpct::device_to_device, handle);
  // CHECK-NEXT: status = (mkl::blas::trmm(handle, (mkl::side)1, (((int)1)==0?(mkl::uplo::lower):(mkl::uplo::upper)), (((int)transpose_ct3)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct3)), (mkl::diag)1, m_ct5, n_ct6, std::complex<double>((&alpha_Z)->x(),(&alpha_Z)->y()), buffer_ct{{[0-9]+}}, lda,  buffer_ct{{[0-9]+}}, ld_ct13), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: {
  // CHECK-NEXT: auto ptr_ct8 = A_Z;
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(ptr_ct8);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto ptr_ct12 = C_Z;
  // CHECK-NEXT: auto allocation_ct{{[0-9]+}} = dpct::mem_mgr::instance().translate_ptr(ptr_ct12);
  // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct{{[0-9]+}} = allocation_ct{{[0-9]+}}.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct{{[0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT: auto ld_ct13 = ldc; auto m_ct5 = m; auto n_ct6 = n;
  // CHECK-NEXT: dpct::matrix_mem_copy(ptr_ct12, B_Z, ld_ct13, ldb, m_ct5, n_ct6, dpct::device_to_device, handle);
  // CHECK-NEXT: mkl::blas::trmm(handle, mkl::side::left, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::nonunit, m_ct5, n_ct6, std::complex<double>((&alpha_Z)->x(),(&alpha_Z)->y()), buffer_ct{{[0-9]+}}, lda,  buffer_ct{{[0-9]+}}, ld_ct13);
  // CHECK-NEXT: }
  status = cublasZtrmm(handle, (cublasSideMode_t)1, (cublasFillMode_t)1, (cublasOperation_t)2, (cublasDiagType_t)1, m, n, &alpha_Z, A_Z, lda, B_Z, ldb, C_Z, ldc);
  cublasZtrmm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &alpha_Z, A_Z, lda, B_Z, ldb, C_Z, ldc);

}
