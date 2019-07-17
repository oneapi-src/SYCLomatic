// RUN: syclct -out-root %T %s  -- -x cuda --cuda-host-only --cuda-path="%cuda-path"
// RUN: FileCheck --input-file %T/cublasTsyrkx.sycl.cpp --match-full-lines %s
#include <cstdio>
#include <cublas.h>
#include <cuda_runtime.h>

//CHECK: #define macro_a (mkl::transpose)1
#define macro_a (cublasOperation_t)1

//CHECK: #define macro_b (mkl::uplo)1
#define macro_b (cublasFillMode_t)1

cublasFillMode_t foo(){
  return CUBLAS_FILL_MODE_LOWER;
}

cublasOperation_t bar(){
  return CUBLAS_OP_T;
}

int main() {
  int n = 275;
  int k = 275;
  int lda = 1;
  int ldb = 1;
  int ldc = 1;

  float alpha_s = 1;
  float beta_s = 1;

  double alpha_d = 1;
  double beta_d = 1;

  cublasHandle_t handle;
  cublasStatus_t status;

  float* A_s=0;
  float* B_s=0;
  float* C_s=0;

  double* A_d=0;
  double* B_d=0;
  double* C_d=0;

  int trans0 = 0;
  int trans1 = 1;
  int fill0 = 0;
  int fill1 = 1;

  //CHECK: /*
  //CHECK-NEXT: SYCLCT1003:0: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  //CHECK-NEXT: */
  //CHECK-NEXT: {
  //CHECK-NEXT: auto transpose_ct_2 = trans0;
  //CHECK-NEXT: auto A_s_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_s);
  //CHECK-NEXT: cl::sycl::buffer<float,1> A_s_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_s_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(A_s_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  //CHECK-NEXT: auto B_s_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_s);
  //CHECK-NEXT: cl::sycl::buffer<float,1> B_s_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_s_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(B_s_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  //CHECK-NEXT: auto C_s_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_s);
  //CHECK-NEXT: cl::sycl::buffer<float,1> C_s_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_s_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(C_s_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  //CHECK-NEXT: status = (mkl::sgemmt(handle, (((int)fill0)==0?(mkl::uplo::lower):(mkl::uplo::upper)), (((int)transpose_ct_2)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct_2)), ((((int)transpose_ct_2)==0)?(mkl::transpose::trans):(mkl::transpose::nontrans)), n, k, *(&alpha_s), A_s_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_s_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, *(&beta_s), C_s_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc), 0);
  //CHECK-NEXT: }
  //CHECK-NEXT: {
  //CHECK-NEXT: auto transpose_ct_2 = trans1;
  //CHECK-NEXT: auto A_s_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_s);
  //CHECK-NEXT: cl::sycl::buffer<float,1> A_s_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_s_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(A_s_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  //CHECK-NEXT: auto B_s_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_s);
  //CHECK-NEXT: cl::sycl::buffer<float,1> B_s_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_s_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(B_s_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  //CHECK-NEXT: auto C_s_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_s);
  //CHECK-NEXT: cl::sycl::buffer<float,1> C_s_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_s_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<float, 1>(cl::sycl::range<1>(C_s_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(float)));
  //CHECK-NEXT: mkl::sgemmt(handle, (((int)fill1)==0?(mkl::uplo::lower):(mkl::uplo::upper)), (((int)transpose_ct_2)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct_2)), ((((int)transpose_ct_2)==0)?(mkl::transpose::trans):(mkl::transpose::nontrans)), n, k, *(&alpha_s), A_s_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_s_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, *(&beta_s), C_s_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  //CHECK-NEXT: }
  status = cublasSsyrkx(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, n, k, &alpha_s, A_s, lda, B_s, ldb, &beta_s, C_s, ldc);
  cublasSsyrkx(handle, (cublasFillMode_t)fill1, (cublasOperation_t)trans1, n, k, &alpha_s, A_s, lda, B_s, ldb, &beta_s, C_s, ldc);

  //CHECK: /*
  //CHECK-NEXT: SYCLCT1003:1: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
  //CHECK-NEXT: */
  //CHECK-NEXT: {
  //CHECK-NEXT: auto transpose_ct_2 = 0;
  //CHECK-NEXT: auto A_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_d);
  //CHECK-NEXT: cl::sycl::buffer<double,1> A_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(A_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  //CHECK-NEXT: auto B_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_d);
  //CHECK-NEXT: cl::sycl::buffer<double,1> B_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(B_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  //CHECK-NEXT: auto C_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_d);
  //CHECK-NEXT: cl::sycl::buffer<double,1> C_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(C_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  //CHECK-NEXT: status = (mkl::dgemmt(handle, (((int)0)==0?(mkl::uplo::lower):(mkl::uplo::upper)), (((int)transpose_ct_2)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct_2)), ((((int)transpose_ct_2)==0)?(mkl::transpose::trans):(mkl::transpose::nontrans)), n, k, *(&alpha_d), A_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, *(&beta_d), C_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc), 0);
  //CHECK-NEXT: }
  //CHECK-NEXT: {
  //CHECK-NEXT: auto transpose_ct_2 = 1;
  //CHECK-NEXT: auto A_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_d);
  //CHECK-NEXT: cl::sycl::buffer<double,1> A_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(A_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  //CHECK-NEXT: auto B_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_d);
  //CHECK-NEXT: cl::sycl::buffer<double,1> B_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(B_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  //CHECK-NEXT: auto C_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_d);
  //CHECK-NEXT: cl::sycl::buffer<double,1> C_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(C_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  //CHECK-NEXT: mkl::dgemmt(handle, (((int)1)==0?(mkl::uplo::lower):(mkl::uplo::upper)), (((int)transpose_ct_2)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct_2)), ((((int)transpose_ct_2)==0)?(mkl::transpose::trans):(mkl::transpose::nontrans)), n, k, *(&alpha_d), A_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, *(&beta_d), C_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  //CHECK-NEXT: }
  status = cublasDsyrkx(handle, (cublasFillMode_t)0, (cublasOperation_t)0, n, k, &alpha_d, A_d, lda, B_d, ldb, &beta_d, C_d, ldc);
  cublasDsyrkx(handle, (cublasFillMode_t)1, (cublasOperation_t)1, n, k, &alpha_d, A_d, lda, B_d, ldb, &beta_d, C_d, ldc);


  //CHECK: {
  //CHECK-NEXT: auto transpose_ct_2 = macro_a;
  //CHECK-NEXT: auto A_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_d);
  //CHECK-NEXT: cl::sycl::buffer<double,1> A_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(A_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  //CHECK-NEXT: auto B_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_d);
  //CHECK-NEXT: cl::sycl::buffer<double,1> B_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(B_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  //CHECK-NEXT: auto C_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_d);
  //CHECK-NEXT: cl::sycl::buffer<double,1> C_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(C_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  //CHECK-NEXT: mkl::dgemmt(handle, foo(), (((int)transpose_ct_2)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct_2)), ((((int)transpose_ct_2)==0)?(mkl::transpose::trans):(mkl::transpose::nontrans)), n, k, *(&alpha_d), A_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, *(&beta_d), C_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  //CHECK-NEXT: }


  cublasDsyrkx(handle, foo(), macro_a, n, k, &alpha_d, A_d, lda, B_d, ldb, &beta_d, C_d, ldc);


  //CHECK: {
  //CHECK-NEXT: auto transpose_ct_2 = bar();
  //CHECK-NEXT: auto A_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_d);
  //CHECK-NEXT: cl::sycl::buffer<double,1> A_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(A_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  //CHECK-NEXT: auto B_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_d);
  //CHECK-NEXT: cl::sycl::buffer<double,1> B_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(B_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  //CHECK-NEXT: auto C_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_d);
  //CHECK-NEXT: cl::sycl::buffer<double,1> C_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(C_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  //CHECK-NEXT: mkl::dgemmt(handle, (((int)macro_b)==0?(mkl::uplo::lower):(mkl::uplo::upper)), transpose_ct_2, ((transpose_ct_2)==(mkl::transpose::nontrans))?(mkl::transpose::trans):(mkl::transpose::nontrans), n, k, *(&alpha_d), A_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, *(&beta_d), C_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  //CHECK-NEXT: }
  cublasDsyrkx(handle, macro_b, bar(), n, k, &alpha_d, A_d, lda, B_d, ldb, &beta_d, C_d, ldc);


  //CHECK: {
  //CHECK-NEXT: auto transpose_ct_2 = mkl::transpose::trans;
  //CHECK-NEXT: auto A_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(A_d);
  //CHECK-NEXT: cl::sycl::buffer<double,1> A_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = A_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(A_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  //CHECK-NEXT: auto B_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(B_d);
  //CHECK-NEXT: cl::sycl::buffer<double,1> B_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = B_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(B_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  //CHECK-NEXT: auto C_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}} = syclct::memory_manager::get_instance().translate_ptr(C_d);
  //CHECK-NEXT: cl::sycl::buffer<double,1> C_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}} = C_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.buffer.reinterpret<double, 1>(cl::sycl::range<1>(C_d_{{[0-9]+}}_allocation_{{[0-9a-z]+}}.size/sizeof(double)));
  //CHECK-NEXT: mkl::dgemmt(handle, mkl::uplo::lower, transpose_ct_2, ((transpose_ct_2)==(mkl::transpose::nontrans))?(mkl::transpose::trans):(mkl::transpose::nontrans), n, k, *(&alpha_d), A_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, lda, B_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldb, *(&beta_d), C_d_{{[0-9]+}}_buffer_{{[0-9a-z]+}}, ldc);
  //CHECK-NEXT: }
  cublasDsyrkx(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, n, k, &alpha_d, A_d, lda, B_d, ldb, &beta_d, C_d, ldc);

  return 0;
}
