// RUN: syclct -out-root %T %s  -- -x cuda --cuda-host-only --cuda-path="%cuda-path"
// RUN: FileCheck --input-file %T/cublasTtrmmLegacy.sycl.cpp --match-full-lines %s
#include <cstdio>
#include <cublas.h>
#include <cuda_runtime.h>

int main(){
  cublasStatus_t status;
  cublasHandle_t handle;
  int n = 275;
  int m = 275;
  int lda = 275;
  int ldb = 275;
  const float *A_S = 0;
  float *B_S = 0;
  float alpha_S = 1.0f;
  const double *A_D = 0;
  double *B_D = 0;
  double alpha_D = 1.0;
  const cuComplex *A_C = 0;
  cuComplex *B_C = 0;
  cuComplex alpha_C = make_cuComplex(1.0f,0.0f);
  const cuDoubleComplex *A_Z = 0;
  cuDoubleComplex *B_Z = 0;
  cuDoubleComplex alpha_Z = make_cuDoubleComplex(1.0,0.0);


  //Legacy
  // CHECK: {
  // CHECK-NEXT: auto sidemode_ct_0 = 'L';
  // CHECK-NEXT: auto fillmode_ct_1 = 'U';
  // CHECK-NEXT: auto transpose_ct_2 = 'N';
  // CHECK-NEXT: auto diagtype_ct_3 = 'N';
  // CHECK-NEXT: auto A_S_7_allocation_89e = syclct::memory_manager::get_instance().translate_ptr(A_S);
  // CHECK-NEXT:cl::sycl::buffer<float,1> A_S_7_buffer_89e = A_S_7_allocation_89e.buffer.reinterpret<float, 1>(cl::sycl::range<1>(A_S_7_allocation_89e.size/sizeof(float)));
  // CHECK-NEXT:auto B_S_9_allocation_f58 = syclct::memory_manager::get_instance().translate_ptr(B_S);
  // CHECK-NEXT:cl::sycl::buffer<float,1> B_S_9_buffer_f58 = B_S_9_allocation_f58.buffer.reinterpret<float, 1>(cl::sycl::range<1>(B_S_9_allocation_f58.size/sizeof(float)));
  // CHECK-NEXT:mkl::strmm(syclct::get_default_queue(), (((sidemode_ct_0)=='L'||(sidemode_ct_0)=='l')?(mkl::side::left):(mkl::side::right)), (((fillmode_ct_1)=='L'||(fillmode_ct_1)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct_2)=='N'||(transpose_ct_2)=='n')?(mkl::transpose::nontrans):(((transpose_ct_2)=='T'||(transpose_ct_2)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), (((diagtype_ct_3)=='N'||(diagtype_ct_3)=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), m, n, alpha_S, A_S_7_buffer_89e, lda, B_S_9_buffer_f58, ldb);
  // CHECK-NEXT:}
  cublasStrmm('L', 'U', 'N', 'N', m, n, alpha_S, A_S, lda, B_S, ldb);

  // CHECK: {
  // CHECK-NEXT: auto sidemode_ct_0 = 'L';
  // CHECK-NEXT: auto fillmode_ct_1 = 'U';
  // CHECK-NEXT: auto transpose_ct_2 = 'N';
  // CHECK-NEXT: auto diagtype_ct_3 = 'N';
  // CHECK-NEXT:auto A_D_7_allocation_382 = syclct::memory_manager::get_instance().translate_ptr(A_D);
  // CHECK-NEXT:cl::sycl::buffer<double,1> A_D_7_buffer_382 = A_D_7_allocation_382.buffer.reinterpret<double, 1>(cl::sycl::range<1>(A_D_7_allocation_382.size/sizeof(double)));
  // CHECK-NEXT:auto B_D_9_allocation_24e = syclct::memory_manager::get_instance().translate_ptr(B_D);
  // CHECK-NEXT:cl::sycl::buffer<double,1> B_D_9_buffer_24e = B_D_9_allocation_24e.buffer.reinterpret<double, 1>(cl::sycl::range<1>(B_D_{{[0-9]+}}_allocation_{{[a-z0-9]+}}.size/sizeof(double)));
  // CHECK-NEXT:mkl::dtrmm(syclct::get_default_queue(), (((sidemode_ct_0)=='L'||(sidemode_ct_0)=='l')?(mkl::side::left):(mkl::side::right)), (((fillmode_ct_1)=='L'||(fillmode_ct_1)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct_2)=='N'||(transpose_ct_2)=='n')?(mkl::transpose::nontrans):(((transpose_ct_2)=='T'||(transpose_ct_2)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), (((diagtype_ct_3)=='N'||(diagtype_ct_3)=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), m, n, alpha_D, A_D_{{[0-9]+}}_buffer_{{[a-z0-9]+}}, lda, B_D_{{[0-9]+}}_buffer_{{[a-z0-9]+}}, ldb);
  // CHECK-NEXT:}
  cublasDtrmm('L', 'U', 'N', 'N', m, n, alpha_D, A_D, lda, B_D, ldb);

  // CHECK: {
  // CHECK-NEXT: auto sidemode_ct_0 = 'L';
  // CHECK-NEXT: auto fillmode_ct_1 = 'U';
  // CHECK-NEXT: auto transpose_ct_2 = 'N';
  // CHECK-NEXT: auto diagtype_ct_3 = 'N';
  // CHECK-NEXT:auto A_C_{{[0-9]+}}_allocation_{{[a-z0-9]+}} = syclct::memory_manager::get_instance().translate_ptr(A_C);
  // CHECK-NEXT:cl::sycl::buffer<std::complex<float>,1> A_C_{{[0-9]+}}_buffer_{{[a-z0-9]+}} = A_C_{{[0-9]+}}_allocation_{{[a-z0-9]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(A_C_{{[0-9]+}}_allocation_{{[a-z0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT:auto B_C_{{[0-9]+}}_allocation_{{[a-z0-9]+}} = syclct::memory_manager::get_instance().translate_ptr(B_C);
  // CHECK-NEXT:cl::sycl::buffer<std::complex<float>,1> B_C_{{[0-9]+}}_buffer_{{[a-z0-9]+}} = B_C_{{[0-9]+}}_allocation_{{[a-z0-9]+}}.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(B_C_{{[0-9]+}}_allocation_{{[a-z0-9]+}}.size/sizeof(std::complex<float>)));
  // CHECK-NEXT:mkl::ctrmm(syclct::get_default_queue(), (((sidemode_ct_0)=='L'||(sidemode_ct_0)=='l')?(mkl::side::left):(mkl::side::right)), (((fillmode_ct_1)=='L'||(fillmode_ct_1)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct_2)=='N'||(transpose_ct_2)=='n')?(mkl::transpose::nontrans):(((transpose_ct_2)=='T'||(transpose_ct_2)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), (((diagtype_ct_3)=='N'||(diagtype_ct_3)=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), m, n, std::complex<float>((alpha_C).x(),(alpha_C).y()), A_C_{{[0-9]+}}_buffer_{{[a-z0-9]+}}, lda, B_C_{{[0-9]+}}_buffer_{{[a-z0-9]+}}, ldb);
  // CHECK-NEXT:}
  cublasCtrmm('L', 'U', 'N', 'N', m, n, alpha_C, A_C, lda, B_C, ldb);

  // CHECK: {
  // CHECK-NEXT: auto sidemode_ct_0 = 'L';
  // CHECK-NEXT: auto fillmode_ct_1 = 'U';
  // CHECK-NEXT: auto transpose_ct_2 = 'N';
  // CHECK-NEXT: auto diagtype_ct_3 = 'N';
  // CHECK-NEXT:auto A_Z_{{[0-9]+}}_allocation_{{[a-z0-9]+}} = syclct::memory_manager::get_instance().translate_ptr(A_Z);
  // CHECK-NEXT:cl::sycl::buffer<std::complex<double>,1> A_Z_{{[0-9]+}}_buffer_{{[a-z0-9]+}} = A_Z_{{[0-9]+}}_allocation_{{[a-z0-9]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(A_Z_{{[0-9]+}}_allocation_{{[a-z0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT:auto B_Z_{{[0-9]+}}_allocation_{{[a-z0-9]+}} = syclct::memory_manager::get_instance().translate_ptr(B_Z);
  // CHECK-NEXT:cl::sycl::buffer<std::complex<double>,1> B_Z_{{[0-9]+}}_buffer_{{[a-z0-9]+}} = B_Z_{{[0-9]+}}_allocation_{{[a-z0-9]+}}.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(B_Z_{{[0-9]+}}_allocation_{{[a-z0-9]+}}.size/sizeof(std::complex<double>)));
  // CHECK-NEXT:mkl::ztrmm(syclct::get_default_queue(), (((sidemode_ct_0)=='L'||(sidemode_ct_0)=='l')?(mkl::side::left):(mkl::side::right)), (((fillmode_ct_1)=='L'||(fillmode_ct_1)=='l')?(mkl::uplo::lower):(mkl::uplo::upper)), (((transpose_ct_2)=='N'||(transpose_ct_2)=='n')?(mkl::transpose::nontrans):(((transpose_ct_2)=='T'||(transpose_ct_2)=='t')?(mkl::transpose::nontrans):(mkl::transpose::conjtrans))), (((diagtype_ct_3)=='N'||(diagtype_ct_3)=='n')?(mkl::diag::nonunit):(mkl::diag::unit)), m, n, std::complex<double>((alpha_Z).x(),(alpha_Z).y()), A_Z_{{[0-9]+}}_buffer_{{[a-z0-9]+}}, lda, B_Z_{{[0-9]+}}_buffer_{{[a-z0-9]+}}, ldb);
  // CHECK-NEXT:}
  cublasZtrmm('L', 'U', 'N', 'N', m, n, alpha_Z, A_Z, lda, B_Z, ldb);
}