// RUN: dpct --format-range=none -out-root %T/datatypes_test_part2 %s --cuda-include-path="%cuda-path/include" --extra-arg="-std=c++14" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/datatypes_test_part2/datatypes_test_part2.dp.cpp

#include <iostream>
#include <cuda.h>
#include <cublas.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cusolverDn.h>
#include <cufft.h>

void case_1(void) {
{
// CHECK: sycl::range<3> var1(1, 1, 1);
// CHECK-NEXT: sycl::range<3> *var2;
// CHECK-NEXT: sycl::range<3> &var3 = var1;
// CHECK-NEXT: sycl::range<3> &&var4 = std::move(var1);
dim3 var1;
dim3 *var2;
dim3 &var3 = var1;
dim3 &&var4 = std::move(var1);
}

{
// CHECK: int var1;
// CHECK-NEXT: int *var2;
// CHECK-NEXT: int &var3 = var1;
// CHECK-NEXT: int &&var4 = std::move(var1);
cudaError_t var1;
cudaError_t *var2;
cudaError_t &var3 = var1;
cudaError_t &&var4 = std::move(var1);
}

{
// CHECK: int var1;
// CHECK-NEXT: int *var2;
// CHECK-NEXT: int &var3 = var1;
// CHECK-NEXT: int &&var4 = std::move(var1);
cudaError var1;
cudaError *var2;
cudaError &var3 = var1;
cudaError &&var4 = std::move(var1);
}

{
// CHECK: int var1;
// CHECK-NEXT: int *var2;
// CHECK-NEXT: int &var3 = var1;
// CHECK-NEXT: int &&var4 = std::move(var1);
CUresult var1;
CUresult *var2;
CUresult &var3 = var1;
CUresult &&var4 = std::move(var1);
}

{
// CHECK: dpct::event_ptr var1;
// CHECK-NEXT: dpct::event_ptr *var2;
// CHECK-NEXT: dpct::event_ptr &var3 = var1;
// CHECK-NEXT: dpct::event_ptr &&var4 = std::move(var1);
cudaEvent_t var1;
cudaEvent_t *var2;
cudaEvent_t &var3 = var1;
cudaEvent_t &&var4 = std::move(var1);
}

{
// CHECK: dpct::queue_ptr var1;
// CHECK-NEXT: dpct::queue_ptr *var2;
// CHECK-NEXT: dpct::queue_ptr &var3 = var1;
// CHECK-NEXT: dpct::queue_ptr &&var4 = std::move(var1);
cublasHandle_t var1;
cublasHandle_t *var2;
cublasHandle_t &var3 = var1;
cublasHandle_t &&var4 = std::move(var1);
}

{
// CHECK: int var1;
// CHECK-NEXT: int *var2;
// CHECK-NEXT: int &var3 = var1;
// CHECK-NEXT: int &&var4 = std::move(var1);
cublasStatus_t var1;
cublasStatus_t *var2;
cublasStatus_t &var3 = var1;
cublasStatus_t &&var4 = std::move(var1);
}

{
// CHECK: sycl::float2 var1;
// CHECK-NEXT: sycl::float2 *var2;
// CHECK-NEXT: sycl::float2 &var3 = var1;
// CHECK-NEXT: sycl::float2 &&var4 = std::move(var1);
cuComplex var1;
cuComplex *var2;
cuComplex &var3 = var1;
cuComplex &&var4 = std::move(var1);
}

{
// CHECK: sycl::double2 var1;
// CHECK-NEXT: sycl::double2 *var2;
// CHECK-NEXT: sycl::double2 &var3 = var1;
// CHECK-NEXT: sycl::double2 &&var4 = std::move(var1);
cuDoubleComplex var1;
cuDoubleComplex *var2;
cuDoubleComplex &var3 = var1;
cuDoubleComplex &&var4 = std::move(var1);
}

{
// CHECK: oneapi::mkl::uplo var1;
// CHECK-NEXT: oneapi::mkl::uplo *var2;
// CHECK-NEXT: oneapi::mkl::uplo &var3 = var1;
// CHECK-NEXT: oneapi::mkl::uplo &&var4 = std::move(var1);
cublasFillMode_t var1;
cublasFillMode_t *var2;
cublasFillMode_t &var3 = var1;
cublasFillMode_t &&var4 = std::move(var1);
}

{
// CHECK: oneapi::mkl::diag var1;
// CHECK-NEXT: oneapi::mkl::diag *var2;
// CHECK-NEXT: oneapi::mkl::diag &var3 = var1;
// CHECK-NEXT: oneapi::mkl::diag &&var4 = std::move(var1);
cublasDiagType_t var1;
cublasDiagType_t *var2;
cublasDiagType_t &var3 = var1;
cublasDiagType_t &&var4 = std::move(var1);
}

{
// CHECK: oneapi::mkl::side var1;
// CHECK-NEXT: oneapi::mkl::side *var2;
// CHECK-NEXT: oneapi::mkl::side &var3 = var1;
// CHECK-NEXT: oneapi::mkl::side &&var4 = std::move(var1);
cublasSideMode_t var1;
cublasSideMode_t *var2;
cublasSideMode_t &var3 = var1;
cublasSideMode_t &&var4 = std::move(var1);
}

{
// CHECK: oneapi::mkl::transpose var1;
// CHECK-NEXT: oneapi::mkl::transpose *var2;
// CHECK-NEXT: oneapi::mkl::transpose &var3 = var1;
// CHECK-NEXT: oneapi::mkl::transpose &&var4 = std::move(var1);
cublasOperation_t var1;
cublasOperation_t *var2;
cublasOperation_t &var3 = var1;
cublasOperation_t &&var4 = std::move(var1);
}

{
// CHECK: int var1;
// CHECK-NEXT: int *var2;
// CHECK-NEXT: int &var3 = var1;
// CHECK-NEXT: int &&var4 = std::move(var1);
cublasStatus var1;
cublasStatus *var2;
cublasStatus &var3 = var1;
cublasStatus &&var4 = std::move(var1);
}

{
// CHECK: int var1;
// CHECK-NEXT: int *var2;
// CHECK-NEXT: int &var3 = var1;
// CHECK-NEXT: int &&var4 = std::move(var1);
cusolverStatus_t var1;
cusolverStatus_t *var2;
cusolverStatus_t &var3 = var1;
cusolverStatus_t &&var4 = std::move(var1);
}

{
// CHECK: int64_t var1;
// CHECK-NEXT: int64_t *var2;
// CHECK-NEXT: int64_t &var3 = var1;
// CHECK-NEXT: int64_t &&var4 = std::move(var1);
cusolverEigType_t var1;
cusolverEigType_t *var2;
cusolverEigType_t &var3 = var1;
cusolverEigType_t &&var4 = std::move(var1);
}

{
// CHECK: oneapi::mkl::job var1;
// CHECK-NEXT: oneapi::mkl::job *var2;
// CHECK-NEXT: oneapi::mkl::job &var3 = var1;
// CHECK-NEXT: oneapi::mkl::job &&var4 = std::move(var1);
cusolverEigMode_t var1;
cusolverEigMode_t *var2;
cusolverEigMode_t &var3 = var1;
cusolverEigMode_t &&var4 = std::move(var1);
}

{
// CHECK: int var1;
// CHECK-NEXT: int *var2;
// CHECK-NEXT: int &var3 = var1;
// CHECK-NEXT: int &&var4 = std::move(var1);
curandStatus_t var1;
curandStatus_t *var2;
curandStatus_t &var3 = var1;
curandStatus_t &&var4 = std::move(var1);
}

{
// CHECK: int var1;
// CHECK-NEXT: int *var2;
// CHECK-NEXT: int &var3 = var1;
// CHECK-NEXT: int &&var4 = std::move(var1);
cufftResult_t var1;
cufftResult_t *var2;
cufftResult_t &var3 = var1;
cufftResult_t &&var4 = std::move(var1);
}

{
// CHECK: dpct::queue_ptr var1;
// CHECK-NEXT: dpct::queue_ptr *var2;
// CHECK-NEXT: dpct::queue_ptr &var3 = var1;
// CHECK-NEXT: dpct::queue_ptr &&var4 = std::move(var1);
cudaStream_t var1;
cudaStream_t *var2;
cudaStream_t &var3 = var1;
cudaStream_t &&var4 = std::move(var1);
}

{
// CHECK: sycl::queue *var2;
CUstream_st *var2;
}
}

// case 2
void case_2(void) {
{
// CHECK:  new sycl::range<3>(1, 1, 1);
// CHECK-NEXT:  new sycl::range<3> *();
  new dim3();
  new dim3 *();
}

{
// CHECK:  new int();
// CHECK-NEXT:  new int *();
  new cudaError_t();
  new cudaError_t *();
}

{
// CHECK:  new int();
// CHECK-NEXT:  new int *();
  new cudaError();
  new cudaError *();
}

{
// CHECK:  new int();
// CHECK-NEXT:  new int *();
  new CUresult();
  new CUresult *();
}

{
// CHECK:  new dpct::event_ptr();
// CHECK-NEXT:  new dpct::event_ptr *();
  new cudaEvent_t();
  new cudaEvent_t *();
}

{
// CHECK:  new dpct::queue_ptr();
// CHECK-NEXT:  new dpct::queue_ptr *();
  new cublasHandle_t();
  new cublasHandle_t *();
}

{
// CHECK:  new int();
// CHECK-NEXT:  new int *();
  new cublasStatus_t();
  new cublasStatus_t *();
}

{
// CHECK:  new sycl::float2();
// CHECK-NEXT:  new sycl::float2 *();
  new cuComplex();
  new cuComplex *();
}

{
// CHECK:  new sycl::double2();
// CHECK-NEXT:  new sycl::double2 *();
  new cuDoubleComplex();
  new cuDoubleComplex *();
}

{
// CHECK:  new oneapi::mkl::uplo();
// CHECK-NEXT:  new oneapi::mkl::uplo *();
  new cublasFillMode_t();
  new cublasFillMode_t *();
}

{
// CHECK:  new oneapi::mkl::diag();
// CHECK-NEXT:  new oneapi::mkl::diag *();
  new cublasDiagType_t();
  new cublasDiagType_t *();
}

{
// CHECK:  new oneapi::mkl::side();
// CHECK-NEXT:  new oneapi::mkl::side *();
  new cublasSideMode_t();
  new cublasSideMode_t *();
}

{
// CHECK:  new oneapi::mkl::transpose();
// CHECK-NEXT:  new oneapi::mkl::transpose *();
  new cublasOperation_t();
  new cublasOperation_t *();
}

{
// CHECK:  new int();
// CHECK-NEXT:  new int *();
  new cublasStatus();
  new cublasStatus *();
}

{
// CHECK:  new int();
// CHECK-NEXT:  new int *();
  new cusolverStatus_t();
  new cusolverStatus_t *();
}

{
// CHECK:  new int64_t();
// CHECK-NEXT:  new int64_t *();
  new cusolverEigType_t();
  new cusolverEigType_t *();
}

{
// CHECK:  new oneapi::mkl::job();
// CHECK-NEXT:  new oneapi::mkl::job *();
  new cusolverEigMode_t();
  new cusolverEigMode_t *();
}

{
// CHECK:  new int();
// CHECK-NEXT:  new int *();
  new curandStatus_t();
  new curandStatus_t *();
}

{
// CHECK:  new int();
// CHECK-NEXT:  new int *();
  new cufftResult_t();
  new cufftResult_t *();
}

{
// CHECK:  new dpct::queue_ptr();
// CHECK-NEXT:  new dpct::queue_ptr *();
  new cudaStream_t();
  new cudaStream_t *();
}

{
// CHECK: new sycl::queue *();
  new CUstream_st *();
}
}

// case 3
// CHECK: sycl::range<3> foo0();
// CHECK-NEXT: sycl::range<3> *foo1();
// CHECK-NEXT: sycl::range<3> &foo2();
dim3 foo0();
dim3 *foo1();
dim3 &foo2();

// CHECK: int foo3();
// CHECK-NEXT: int *foo4();
// CHECK-NEXT: int &foo5();
cudaError_t foo3();
cudaError_t *foo4();
cudaError_t &foo5();

// CHECK: int foo6();
// CHECK-NEXT: int *foo7();
// CHECK-NEXT: int &foo8();
cudaError foo6();
cudaError *foo7();
cudaError &foo8();

// CHECK: int foo9();
// CHECK-NEXT: int *foo10();
// CHECK-NEXT: int &foo11();
CUresult foo9();
CUresult *foo10();
CUresult &foo11();

// CHECK: dpct::event_ptr foo12();
// CHECK-NEXT: dpct::event_ptr *foo13();
// CHECK-NEXT: dpct::event_ptr &foo14();
cudaEvent_t foo12();
cudaEvent_t *foo13();
cudaEvent_t &foo14();

// CHECK: dpct::queue_ptr foo15();
// CHECK-NEXT: dpct::queue_ptr *foo16();
// CHECK-NEXT: dpct::queue_ptr &foo17();
cublasHandle_t foo15();
cublasHandle_t *foo16();
cublasHandle_t &foo17();

// CHECK: int foo18();
// CHECK-NEXT: int *foo19();
// CHECK-NEXT: int &foo20();
cublasStatus_t foo18();
cublasStatus_t *foo19();
cublasStatus_t &foo20();

// CHECK: sycl::float2 foo21();
// CHECK-NEXT: sycl::float2 *foo22();
// CHECK-NEXT: sycl::float2 &foo23();
cuComplex foo21();
cuComplex *foo22();
cuComplex &foo23();

// CHECK: sycl::double2 foo24();
// CHECK-NEXT: sycl::double2 *foo25();
// CHECK-NEXT: sycl::double2 &foo26();
cuDoubleComplex foo24();
cuDoubleComplex *foo25();
cuDoubleComplex &foo26();

// CHECK: oneapi::mkl::uplo foo27();
// CHECK-NEXT: oneapi::mkl::uplo *foo28();
// CHECK-NEXT: oneapi::mkl::uplo &foo29();
cublasFillMode_t foo27();
cublasFillMode_t *foo28();
cublasFillMode_t &foo29();

// CHECK: oneapi::mkl::diag foo30();
// CHECK-NEXT: oneapi::mkl::diag *foo31();
// CHECK-NEXT: oneapi::mkl::diag &foo32();
cublasDiagType_t foo30();
cublasDiagType_t *foo31();
cublasDiagType_t &foo32();

// CHECK: oneapi::mkl::side foo33();
// CHECK-NEXT: oneapi::mkl::side *foo34();
// CHECK-NEXT: oneapi::mkl::side &foo35();
cublasSideMode_t foo33();
cublasSideMode_t *foo34();
cublasSideMode_t &foo35();

// CHECK: oneapi::mkl::transpose foo36();
// CHECK-NEXT: oneapi::mkl::transpose *foo37();
// CHECK-NEXT: oneapi::mkl::transpose &foo38();
cublasOperation_t foo36();
cublasOperation_t *foo37();
cublasOperation_t &foo38();

// CHECK: int foo39();
// CHECK-NEXT: int *foo40();
// CHECK-NEXT: int &foo41();
cublasStatus foo39();
cublasStatus *foo40();
cublasStatus &foo41();

// CHECK: int foo42();
// CHECK-NEXT: int *foo43();
// CHECK-NEXT: int &foo44();
cusolverStatus_t foo42();
cusolverStatus_t *foo43();
cusolverStatus_t &foo44();

// CHECK: int64_t foo45();
// CHECK-NEXT: int64_t *foo46();
// CHECK-NEXT: int64_t &foo47();
cusolverEigType_t foo45();
cusolverEigType_t *foo46();
cusolverEigType_t &foo47();

// CHECK: oneapi::mkl::job foo48();
// CHECK-NEXT: oneapi::mkl::job *foo49();
// CHECK-NEXT: oneapi::mkl::job &foo50();
cusolverEigMode_t foo48();
cusolverEigMode_t *foo49();
cusolverEigMode_t &foo50();

// CHECK: int foo51();
// CHECK-NEXT: int *foo52();
// CHECK-NEXT: int &foo53();
curandStatus_t foo51();
curandStatus_t *foo52();
curandStatus_t &foo53();

// CHECK: int foo54();
// CHECK-NEXT: int *foo55();
// CHECK-NEXT: int &foo56();
cufftResult_t foo54();
cufftResult_t *foo55();
cufftResult_t &foo56();

// CHECK: dpct::queue_ptr foo57();
// CHECK-NEXT: dpct::queue_ptr *foo58();
// CHECK-NEXT: dpct::queue_ptr &foo59();
cudaStream_t foo57();
cudaStream_t *foo58();
cudaStream_t &foo59();

// CHECK: sycl::queue foo_1();
// CHECK-NEXT: sycl::queue *foo_2();
CUstream_st foo_1();
CUstream_st *foo_2();


// case 4
template <typename T> struct S {};

// CHECK: template <> struct S<sycl::range<3>> {};
// CHECK-NEXT: template <> struct S<sycl::range<3> *> {};
// CHECK-NEXT: template <> struct S<sycl::range<3> &> {};
// CHECK-NEXT: template <> struct S<sycl::range<3> &&> {};
template <> struct S<dim3> {};
template <> struct S<dim3 *> {};
template <> struct S<dim3 &> {};
template <> struct S<dim3 &&> {};

// CHECK: template <> struct S<int> {};
// CHECK-NEXT: template <> struct S<int *> {};
// CHECK-NEXT: template <> struct S<int &> {};
// CHECK-NEXT: template <> struct S<int &&> {};
template <> struct S<cudaError> {};
template <> struct S<cudaError *> {};
template <> struct S<cudaError &> {};
template <> struct S<cudaError &&> {};

// CHECK: template <> struct S<int> {};
// CHECK-NEXT: template <> struct S<int *> {};
// CHECK-NEXT: template <> struct S<int &> {};
// CHECK-NEXT: template <> struct S<int &&> {};
template <> struct S<CUresult> {};
template <> struct S<CUresult *> {};
template <> struct S<CUresult &> {};
template <> struct S<CUresult &&> {};

// CHECK: template <> struct S<dpct::event_ptr> {};
// CHECK-NEXT: template <> struct S<dpct::event_ptr *> {};
// CHECK-NEXT: template <> struct S<dpct::event_ptr &> {};
// CHECK-NEXT: template <> struct S<dpct::event_ptr &&> {};
template <> struct S<cudaEvent_t> {};
template <> struct S<cudaEvent_t *> {};
template <> struct S<cudaEvent_t &> {};
template <> struct S<cudaEvent_t &&> {};

// CHECK: template <> struct S<dpct::queue_ptr> {};
// CHECK-NEXT: template <> struct S<dpct::queue_ptr *> {};
// CHECK-NEXT: template <> struct S<dpct::queue_ptr &> {};
// CHECK-NEXT: template <> struct S<dpct::queue_ptr &&> {};
template <> struct S<cublasHandle_t> {};
template <> struct S<cublasHandle_t *> {};
template <> struct S<cublasHandle_t &> {};
template <> struct S<cublasHandle_t &&> {};


// CHECK: template <> struct S<sycl::float2> {};
// CHECK-NEXT: template <> struct S<sycl::float2 *> {};
// CHECK-NEXT: template <> struct S<sycl::float2 &> {};
// CHECK-NEXT: template <> struct S<sycl::float2 &&> {};
template <> struct S<cuComplex> {};
template <> struct S<cuComplex *> {};
template <> struct S<cuComplex &> {};
template <> struct S<cuComplex &&> {};

// CHECK: template <> struct S<sycl::double2> {};
// CHECK-NEXT: template <> struct S<sycl::double2 *> {};
// CHECK-NEXT: template <> struct S<sycl::double2 &> {};
// CHECK-NEXT: template <> struct S<sycl::double2 &&> {};
template <> struct S<cuDoubleComplex> {};
template <> struct S<cuDoubleComplex *> {};
template <> struct S<cuDoubleComplex &> {};
template <> struct S<cuDoubleComplex &&> {};

// CHECK: template <> struct S<oneapi::mkl::uplo> {};
// CHECK-NEXT: template <> struct S<oneapi::mkl::uplo *> {};
// CHECK-NEXT: template <> struct S<oneapi::mkl::uplo &> {};
// CHECK-NEXT: template <> struct S<oneapi::mkl::uplo &&> {};
template <> struct S<cublasFillMode_t> {};
template <> struct S<cublasFillMode_t *> {};
template <> struct S<cublasFillMode_t &> {};
template <> struct S<cublasFillMode_t &&> {};

// CHECK: template <> struct S<oneapi::mkl::diag> {};
// CHECK-NEXT: template <> struct S<oneapi::mkl::diag *> {};
// CHECK-NEXT: template <> struct S<oneapi::mkl::diag &> {};
// CHECK-NEXT: template <> struct S<oneapi::mkl::diag &&> {};
template <> struct S<cublasDiagType_t> {};
template <> struct S<cublasDiagType_t *> {};
template <> struct S<cublasDiagType_t &> {};
template <> struct S<cublasDiagType_t &&> {};

// CHECK: template <> struct S<oneapi::mkl::side> {};
// CHECK-NEXT: template <> struct S<oneapi::mkl::side *> {};
// CHECK-NEXT: template <> struct S<oneapi::mkl::side &> {};
// CHECK-NEXT: template <> struct S<oneapi::mkl::side &&> {};
template <> struct S<cublasSideMode_t> {};
template <> struct S<cublasSideMode_t *> {};
template <> struct S<cublasSideMode_t &> {};
template <> struct S<cublasSideMode_t &&> {};

// CHECK: template <> struct S<oneapi::mkl::transpose> {};
// CHECK-NEXT: template <> struct S<oneapi::mkl::transpose *> {};
// CHECK-NEXT: template <> struct S<oneapi::mkl::transpose &> {};
// CHECK-NEXT: template <> struct S<oneapi::mkl::transpose &&> {};
template <> struct S<cublasOperation_t> {};
template <> struct S<cublasOperation_t *> {};
template <> struct S<cublasOperation_t &> {};
template <> struct S<cublasOperation_t &&> {};

// CHECK: template <> struct S<int> {};
// CHECK-NEXT: template <> struct S<int *> {};
// CHECK-NEXT: template <> struct S<int &> {};
// CHECK-NEXT: template <> struct S<int &&> {};
template <> struct S<cublasStatus> {};
template <> struct S<cublasStatus *> {};
template <> struct S<cublasStatus &> {};
template <> struct S<cublasStatus &&> {};

// CHECK: template <> struct S<int> {};
// CHECK-NEXT: template <> struct S<int *> {};
// CHECK-NEXT: template <> struct S<int &> {};
// CHECK-NEXT: template <> struct S<int &&> {};
template <> struct S<cusolverStatus_t> {};
template <> struct S<cusolverStatus_t *> {};
template <> struct S<cusolverStatus_t &> {};
template <> struct S<cusolverStatus_t &&> {};

// CHECK: template <> struct S<int64_t> {};
// CHECK-NEXT: template <> struct S<int64_t *> {};
// CHECK-NEXT: template <> struct S<int64_t &> {};
// CHECK-NEXT: template <> struct S<int64_t &&> {};
template <> struct S<cusolverEigType_t> {};
template <> struct S<cusolverEigType_t *> {};
template <> struct S<cusolverEigType_t &> {};
template <> struct S<cusolverEigType_t &&> {};

// CHECK: template <> struct S<oneapi::mkl::job> {};
// CHECK-NEXT: template <> struct S<oneapi::mkl::job *> {};
// CHECK-NEXT: template <> struct S<oneapi::mkl::job &> {};
// CHECK-NEXT: template <> struct S<oneapi::mkl::job &&> {};
template <> struct S<cusolverEigMode_t> {};
template <> struct S<cusolverEigMode_t *> {};
template <> struct S<cusolverEigMode_t &> {};
template <> struct S<cusolverEigMode_t &&> {};

// CHECK: template <> struct S<int> {};
// CHECK-NEXT: template <> struct S<int *> {};
// CHECK-NEXT: template <> struct S<int &> {};
// CHECK-NEXT: template <> struct S<int &&> {};
template <> struct S<curandStatus_t> {};
template <> struct S<curandStatus_t *> {};
template <> struct S<curandStatus_t &> {};
template <> struct S<curandStatus_t &&> {};

// CHECK: template <> struct S<int> {};
// CHECK-NEXT: template <> struct S<int *> {};
// CHECK-NEXT: template <> struct S<int &> {};
// CHECK-NEXT: template <> struct S<int &&> {};
template <> struct S<cufftResult_t> {};
template <> struct S<cufftResult_t *> {};
template <> struct S<cufftResult_t &> {};
template <> struct S<cufftResult_t &&> {};

// CHECK: template <> struct S<dpct::queue_ptr> {};
// CHECK-NEXT: template <> struct S<dpct::queue_ptr *> {};
// CHECK-NEXT: template <> struct S<dpct::queue_ptr &> {};
// CHECK-NEXT: template <> struct S<dpct::queue_ptr &&> {};
template <> struct S<cudaStream_t> {};
template <> struct S<cudaStream_t *> {};
template <> struct S<cudaStream_t &> {};
template <> struct S<cudaStream_t &&> {};


// case 5
template <typename T> void template_foo() {}
void case_5(){

// CHECK: template_foo<sycl::range<3>>();
// CHECK-NEXT: template_foo<sycl::range<3> *>();
// CHECK-NEXT: template_foo<sycl::range<3> &>();
// CHECK-NEXT: template_foo<sycl::range<3> &&>();
template_foo<dim3>();
template_foo<dim3 *>();
template_foo<dim3 &>();
template_foo<dim3 &&>();

// CHECK: template_foo<int>();
// CHECK-NEXT: template_foo<int *>();
// CHECK-NEXT: template_foo<int &>();
// CHECK-NEXT: template_foo<int &&>();
template_foo<cudaError_t>();
template_foo<cudaError_t *>();
template_foo<cudaError_t &>();
template_foo<cudaError_t &&>();

// CHECK: template_foo<int>();
// CHECK-NEXT: template_foo<int *>();
// CHECK-NEXT: template_foo<int &>();
// CHECK-NEXT: template_foo<int &&>();
template_foo<cudaError>();
template_foo<cudaError *>();
template_foo<cudaError &>();
template_foo<cudaError &&>();

// CHECK: template_foo<int>();
// CHECK-NEXT: template_foo<int *>();
// CHECK-NEXT: template_foo<int &>();
// CHECK-NEXT: template_foo<int &&>();
template_foo<CUresult>();
template_foo<CUresult *>();
template_foo<CUresult &>();
template_foo<CUresult &&>();

// CHECK: template_foo<dpct::event_ptr>();
// CHECK-NEXT: template_foo<dpct::event_ptr *>();
// CHECK-NEXT: template_foo<dpct::event_ptr &>();
// CHECK-NEXT: template_foo<dpct::event_ptr &&>();
template_foo<cudaEvent_t>();
template_foo<cudaEvent_t *>();
template_foo<cudaEvent_t &>();
template_foo<cudaEvent_t &&>();

// CHECK: template_foo<dpct::queue_ptr>();
// CHECK-NEXT: template_foo<dpct::queue_ptr *>();
// CHECK-NEXT: template_foo<dpct::queue_ptr &>();
// CHECK-NEXT: template_foo<dpct::queue_ptr &&>();
template_foo<cublasHandle_t>();
template_foo<cublasHandle_t *>();
template_foo<cublasHandle_t &>();
template_foo<cublasHandle_t &&>();

// CHECK: template_foo<int>();
// CHECK-NEXT: template_foo<int *>();
// CHECK-NEXT: template_foo<int &>();
// CHECK-NEXT: template_foo<int &&>();
template_foo<cublasStatus_t>();
template_foo<cublasStatus_t *>();
template_foo<cublasStatus_t &>();
template_foo<cublasStatus_t &&>();

// CHECK: template_foo<sycl::float2>();
// CHECK-NEXT: template_foo<sycl::float2 *>();
// CHECK-NEXT: template_foo<sycl::float2 &>();
// CHECK-NEXT: template_foo<sycl::float2 &&>();
template_foo<cuComplex>();
template_foo<cuComplex *>();
template_foo<cuComplex &>();
template_foo<cuComplex &&>();

// CHECK: template_foo<sycl::double2>();
// CHECK-NEXT: template_foo<sycl::double2 *>();
// CHECK-NEXT: template_foo<sycl::double2 &>();
// CHECK-NEXT: template_foo<sycl::double2 &&>();
template_foo<cuDoubleComplex>();
template_foo<cuDoubleComplex *>();
template_foo<cuDoubleComplex &>();
template_foo<cuDoubleComplex &&>();

// CHECK: template_foo<oneapi::mkl::uplo>();
// CHECK-NEXT: template_foo<oneapi::mkl::uplo *>();
// CHECK-NEXT: template_foo<oneapi::mkl::uplo &>();
// CHECK-NEXT: template_foo<oneapi::mkl::uplo &&>();
template_foo<cublasFillMode_t>();
template_foo<cublasFillMode_t *>();
template_foo<cublasFillMode_t &>();
template_foo<cublasFillMode_t &&>();

// CHECK: template_foo<oneapi::mkl::diag>();
// CHECK-NEXT: template_foo<oneapi::mkl::diag *>();
// CHECK-NEXT: template_foo<oneapi::mkl::diag &>();
// CHECK-NEXT: template_foo<oneapi::mkl::diag &&>();
template_foo<cublasDiagType_t>();
template_foo<cublasDiagType_t *>();
template_foo<cublasDiagType_t &>();
template_foo<cublasDiagType_t &&>();

// CHECK: template_foo<oneapi::mkl::side>();
// CHECK-NEXT: template_foo<oneapi::mkl::side *>();
// CHECK-NEXT: template_foo<oneapi::mkl::side &>();
// CHECK-NEXT: template_foo<oneapi::mkl::side &&>();
template_foo<cublasSideMode_t>();
template_foo<cublasSideMode_t *>();
template_foo<cublasSideMode_t &>();
template_foo<cublasSideMode_t &&>();

// CHECK: template_foo<oneapi::mkl::transpose>();
// CHECK-NEXT: template_foo<oneapi::mkl::transpose *>();
// CHECK-NEXT: template_foo<oneapi::mkl::transpose &>();
// CHECK-NEXT: template_foo<oneapi::mkl::transpose &&>();
template_foo<cublasOperation_t>();
template_foo<cublasOperation_t *>();
template_foo<cublasOperation_t &>();
template_foo<cublasOperation_t &&>();

// CHECK: template_foo<int>();
// CHECK-NEXT: template_foo<int *>();
// CHECK-NEXT: template_foo<int &>();
// CHECK-NEXT: template_foo<int &&>();
template_foo<cublasStatus>();
template_foo<cublasStatus *>();
template_foo<cublasStatus &>();
template_foo<cublasStatus &&>();

// CHECK: template_foo<int>();
// CHECK-NEXT: template_foo<int *>();
// CHECK-NEXT: template_foo<int &>();
// CHECK-NEXT: template_foo<int &&>();
template_foo<cusolverStatus_t>();
template_foo<cusolverStatus_t *>();
template_foo<cusolverStatus_t &>();
template_foo<cusolverStatus_t &&>();

// CHECK: template_foo<int64_t>();
// CHECK-NEXT: template_foo<int64_t *>();
// CHECK-NEXT: template_foo<int64_t &>();
// CHECK-NEXT: template_foo<int64_t &&>();
template_foo<cusolverEigType_t>();
template_foo<cusolverEigType_t *>();
template_foo<cusolverEigType_t &>();
template_foo<cusolverEigType_t &&>();

// CHECK: template_foo<oneapi::mkl::job>();
// CHECK-NEXT: template_foo<oneapi::mkl::job *>();
// CHECK-NEXT: template_foo<oneapi::mkl::job &>();
// CHECK-NEXT: template_foo<oneapi::mkl::job &&>();
template_foo<cusolverEigMode_t>();
template_foo<cusolverEigMode_t *>();
template_foo<cusolverEigMode_t &>();
template_foo<cusolverEigMode_t &&>();

// CHECK: template_foo<int>();
// CHECK-NEXT: template_foo<int *>();
// CHECK-NEXT: template_foo<int &>();
// CHECK-NEXT: template_foo<int &&>();
template_foo<curandStatus_t>();
template_foo<curandStatus_t *>();
template_foo<curandStatus_t &>();
template_foo<curandStatus_t &&>();

// CHECK: template_foo<int>();
// CHECK-NEXT: template_foo<int *>();
// CHECK-NEXT: template_foo<int &>();
// CHECK-NEXT: template_foo<int &&>();
template_foo<cufftResult_t>();
template_foo<cufftResult_t *>();
template_foo<cufftResult_t &>();
template_foo<cufftResult_t &&>();

// CHECK: template_foo<dpct::queue_ptr>();
// CHECK-NEXT: template_foo<dpct::queue_ptr *>();
// CHECK-NEXT: template_foo<dpct::queue_ptr &>();
// CHECK-NEXT: template_foo<dpct::queue_ptr &&>();
template_foo<cudaStream_t>();
template_foo<cudaStream_t *>();
template_foo<cudaStream_t &>();
template_foo<cudaStream_t &&>();

// CHECK: template_foo<sycl::queue>();
// CHECK-NEXT: template_foo<sycl::queue *>();
// CHECK-NEXT: template_foo<sycl::queue &>();
// CHECK-NEXT: template_foo<sycl::queue &&>();
template_foo<CUstream_st>();
template_foo<CUstream_st *>();
template_foo<CUstream_st &>();
template_foo<CUstream_st &&>();

}


// case 6
// CHECK: using UT0 = sycl::range<3>;
// CHECK-NEXT: using UT1 = sycl::range<3> *;
// CHECK-NEXT: using UT2 = sycl::range<3> &;
// CHECK-NEXT: using UT3 = sycl::range<3> &&;
using UT0 = dim3;
using UT1 = dim3 *;
using UT2 = dim3 &;
using UT3 = dim3 &&;

// CHECK: using UT4 = int;
// CHECK-NEXT: using UT5 = int *;
// CHECK-NEXT: using UT6 = int &;
// CHECK-NEXT: using UT7 = int &&;
using UT4 = cudaError_t;
using UT5 = cudaError_t *;
using UT6 = cudaError_t &;
using UT7 = cudaError_t &&;

// CHECK: using UT8 = int;
// CHECK-NEXT: using UT9 = int *;
// CHECK-NEXT: using UT10 = int &;
// CHECK-NEXT: using UT11 = int &&;
using UT8 = cudaError;
using UT9 = cudaError *;
using UT10 = cudaError &;
using UT11 = cudaError &&;

// CHECK: using UT12 = int;
// CHECK-NEXT: using UT13 = int *;
// CHECK-NEXT: using UT14 = int &;
// CHECK-NEXT: using UT15 = int &&;
using UT12 = CUresult;
using UT13 = CUresult *;
using UT14 = CUresult &;
using UT15 = CUresult &&;

// CHECK: using UT16 = dpct::event_ptr;
// CHECK-NEXT: using UT17 = dpct::event_ptr *;
// CHECK-NEXT: using UT18 = dpct::event_ptr &;
// CHECK-NEXT: using UT19 = dpct::event_ptr &&;
using UT16 = cudaEvent_t;
using UT17 = cudaEvent_t *;
using UT18 = cudaEvent_t &;
using UT19 = cudaEvent_t &&;

// CHECK: using UT20 = dpct::queue_ptr;
// CHECK-NEXT: using UT21 = dpct::queue_ptr *;
// CHECK-NEXT: using UT22 = dpct::queue_ptr &;
// CHECK-NEXT: using UT23 = dpct::queue_ptr &&;
using UT20 = cublasHandle_t;
using UT21 = cublasHandle_t *;
using UT22 = cublasHandle_t &;
using UT23 = cublasHandle_t &&;

// CHECK: using UT24 = int;
// CHECK-NEXT: using UT25 = int *;
// CHECK-NEXT: using UT26 = int &;
// CHECK-NEXT: using UT27 = int &&;
using UT24 = cublasStatus_t;
using UT25 = cublasStatus_t *;
using UT26 = cublasStatus_t &;
using UT27 = cublasStatus_t &&;

// CHECK: using UT28 = sycl::float2;
// CHECK-NEXT: using UT29 = sycl::float2 *;
// CHECK-NEXT: using UT30 = sycl::float2 &;
// CHECK-NEXT: using UT31 = sycl::float2 &&;
using UT28 = cuComplex;
using UT29 = cuComplex *;
using UT30 = cuComplex &;
using UT31 = cuComplex &&;

// CHECK: using UT32 = sycl::double2;
// CHECK-NEXT: using UT33 = sycl::double2 *;
// CHECK-NEXT: using UT34 = sycl::double2 &;
// CHECK-NEXT: using UT35 = sycl::double2 &&;
using UT32 = cuDoubleComplex;
using UT33 = cuDoubleComplex *;
using UT34 = cuDoubleComplex &;
using UT35 = cuDoubleComplex &&;

// CHECK: using UT36 = oneapi::mkl::uplo;
// CHECK-NEXT: using UT37 = oneapi::mkl::uplo *;
// CHECK-NEXT: using UT38 = oneapi::mkl::uplo &;
// CHECK-NEXT: using UT39 = oneapi::mkl::uplo &&;
using UT36 = cublasFillMode_t;
using UT37 = cublasFillMode_t *;
using UT38 = cublasFillMode_t &;
using UT39 = cublasFillMode_t &&;

// CHECK: using UT40 = oneapi::mkl::diag;
// CHECK-NEXT: using UT41 = oneapi::mkl::diag *;
// CHECK-NEXT: using UT42 = oneapi::mkl::diag &;
// CHECK-NEXT: using UT43 = oneapi::mkl::diag &&;
using UT40 = cublasDiagType_t;
using UT41 = cublasDiagType_t *;
using UT42 = cublasDiagType_t &;
using UT43 = cublasDiagType_t &&;

// CHECK: using UT44 = oneapi::mkl::side;
// CHECK-NEXT: using UT45 = oneapi::mkl::side *;
// CHECK-NEXT: using UT46 = oneapi::mkl::side &;
// CHECK-NEXT: using UT47 = oneapi::mkl::side &&;
using UT44 = cublasSideMode_t;
using UT45 = cublasSideMode_t *;
using UT46 = cublasSideMode_t &;
using UT47 = cublasSideMode_t &&;

// CHECK: using UT48 = oneapi::mkl::transpose;
// CHECK-NEXT: using UT49 = oneapi::mkl::transpose *;
// CHECK-NEXT: using UT50 = oneapi::mkl::transpose &;
// CHECK-NEXT: using UT51 = oneapi::mkl::transpose &&;
using UT48 = cublasOperation_t;
using UT49 = cublasOperation_t *;
using UT50 = cublasOperation_t &;
using UT51 = cublasOperation_t &&;

// CHECK: using UT52 = int;
// CHECK-NEXT: using UT53 = int *;
// CHECK-NEXT: using UT54 = int &;
// CHECK-NEXT: using UT55 = int &&;
using UT52 = cublasStatus;
using UT53 = cublasStatus *;
using UT54 = cublasStatus &;
using UT55 = cublasStatus &&;

// CHECK: using UT56 = int;
// CHECK-NEXT: using UT57 = int *;
// CHECK-NEXT: using UT58 = int &;
// CHECK-NEXT: using UT59 = int &&;
using UT56 = cusolverStatus_t;
using UT57 = cusolverStatus_t *;
using UT58 = cusolverStatus_t &;
using UT59 = cusolverStatus_t &&;

// CHECK: using UT60 = int64_t;
// CHECK-NEXT: using UT61 = int64_t *;
// CHECK-NEXT: using UT62 = int64_t &;
// CHECK-NEXT: using UT63 = int64_t &&;
using UT60 = cusolverEigType_t;
using UT61 = cusolverEigType_t *;
using UT62 = cusolverEigType_t &;
using UT63 = cusolverEigType_t &&;

// CHECK: using UT64 = oneapi::mkl::job;
// CHECK-NEXT: using UT65 = oneapi::mkl::job *;
// CHECK-NEXT: using UT66 = oneapi::mkl::job &;
// CHECK-NEXT: using UT67 = oneapi::mkl::job &&;
using UT64 = cusolverEigMode_t;
using UT65 = cusolverEigMode_t *;
using UT66 = cusolverEigMode_t &;
using UT67 = cusolverEigMode_t &&;

// CHECK: using UT68 = int;
// CHECK-NEXT: using UT69 = int *;
// CHECK-NEXT: using UT70 = int &;
// CHECK-NEXT: using UT71 = int &&;
using UT68 = curandStatus_t;
using UT69 = curandStatus_t *;
using UT70 = curandStatus_t &;
using UT71 = curandStatus_t &&;

// CHECK: using UT72 = int;
// CHECK-NEXT: using UT73 = int *;
// CHECK-NEXT: using UT74 = int &;
// CHECK-NEXT: using UT75 = int &&;
using UT72 = cufftResult_t;
using UT73 = cufftResult_t *;
using UT74 = cufftResult_t &;
using UT75 = cufftResult_t &&;

// CHECK: using UT76 = dpct::queue_ptr;
// CHECK-NEXT: using UT77 = dpct::queue_ptr *;
// CHECK-NEXT: using UT78 = dpct::queue_ptr &;
// CHECK-NEXT: using UT79 = dpct::queue_ptr &&;
using UT76 = cudaStream_t;
using UT77 = cudaStream_t *;
using UT78 = cudaStream_t &;
using UT79 = cudaStream_t &&;

// CHECK: using UT_1 = sycl::queue;
// CHECK-NEXT: using UT_2 = sycl::queue *;
// CHECK-NEXT: using UT_3 = sycl::queue &;
// CHECK-NEXT: using UT_4 = sycl::queue &&;
using UT_1 = CUstream_st;
using UT_2 = CUstream_st *;
using UT_3 = CUstream_st &;
using UT_4 = CUstream_st &&;



// case 7
// CHECK: typedef sycl::range<3> T0;
// CHECK-NEXT: typedef sycl::range<3>* T1;
// CHECK-NEXT: typedef sycl::range<3>& T2;
// CHECK-NEXT: typedef sycl::range<3>&& T3;
typedef dim3 T0;
typedef dim3* T1;
typedef dim3& T2;
typedef dim3&& T3;

// CHECK: typedef int T4;
// CHECK-NEXT: typedef int* T5;
// CHECK-NEXT: typedef int& T6;
// CHECK-NEXT: typedef int&& T7;
typedef cudaError_t T4;
typedef cudaError_t* T5;
typedef cudaError_t& T6;
typedef cudaError_t&& T7;

// CHECK: typedef int T8;
// CHECK-NEXT: typedef int* T9;
// CHECK-NEXT: typedef int& T10;
// CHECK-NEXT: typedef int&& T11;
typedef cudaError T8;
typedef cudaError* T9;
typedef cudaError& T10;
typedef cudaError&& T11;

// CHECK: typedef int T12;
// CHECK-NEXT: typedef int* T13;
// CHECK-NEXT: typedef int& T14;
// CHECK-NEXT: typedef int&& T15;
typedef CUresult T12;
typedef CUresult* T13;
typedef CUresult& T14;
typedef CUresult&& T15;

// CHECK: typedef dpct::event_ptr T16;
// CHECK-NEXT: typedef dpct::event_ptr* T17;
// CHECK-NEXT: typedef dpct::event_ptr& T18;
// CHECK-NEXT: typedef dpct::event_ptr&& T19;
typedef cudaEvent_t T16;
typedef cudaEvent_t* T17;
typedef cudaEvent_t& T18;
typedef cudaEvent_t&& T19;

// CHECK: typedef dpct::queue_ptr T20;
// CHECK-NEXT: typedef dpct::queue_ptr* T21;
// CHECK-NEXT: typedef dpct::queue_ptr& T22;
// CHECK-NEXT: typedef dpct::queue_ptr&& T23;
typedef cublasHandle_t T20;
typedef cublasHandle_t* T21;
typedef cublasHandle_t& T22;
typedef cublasHandle_t&& T23;

// CHECK: typedef int T24;
// CHECK-NEXT: typedef int* T25;
// CHECK-NEXT: typedef int& T26;
// CHECK-NEXT: typedef int&& T27;
typedef cublasStatus_t T24;
typedef cublasStatus_t* T25;
typedef cublasStatus_t& T26;
typedef cublasStatus_t&& T27;

// CHECK: typedef sycl::float2 T28;
// CHECK-NEXT: typedef sycl::float2* T29;
// CHECK-NEXT: typedef sycl::float2& T30;
// CHECK-NEXT: typedef sycl::float2&& T31;
typedef cuComplex T28;
typedef cuComplex* T29;
typedef cuComplex& T30;
typedef cuComplex&& T31;

// CHECK: typedef sycl::double2 T32;
// CHECK-NEXT: typedef sycl::double2* T33;
// CHECK-NEXT: typedef sycl::double2& T34;
// CHECK-NEXT: typedef sycl::double2&& T35;
typedef cuDoubleComplex T32;
typedef cuDoubleComplex* T33;
typedef cuDoubleComplex& T34;
typedef cuDoubleComplex&& T35;

// CHECK: typedef oneapi::mkl::uplo T36;
// CHECK-NEXT: typedef oneapi::mkl::uplo* T37;
// CHECK-NEXT: typedef oneapi::mkl::uplo& T38;
// CHECK-NEXT: typedef oneapi::mkl::uplo&& T39;
typedef cublasFillMode_t T36;
typedef cublasFillMode_t* T37;
typedef cublasFillMode_t& T38;
typedef cublasFillMode_t&& T39;

// CHECK: typedef oneapi::mkl::diag T40;
// CHECK-NEXT: typedef oneapi::mkl::diag* T41;
// CHECK-NEXT: typedef oneapi::mkl::diag& T42;
// CHECK-NEXT: typedef oneapi::mkl::diag&& T43;
typedef cublasDiagType_t T40;
typedef cublasDiagType_t* T41;
typedef cublasDiagType_t& T42;
typedef cublasDiagType_t&& T43;

// CHECK: typedef oneapi::mkl::side T44;
// CHECK-NEXT: typedef oneapi::mkl::side* T45;
// CHECK-NEXT: typedef oneapi::mkl::side& T46;
// CHECK-NEXT: typedef oneapi::mkl::side&& T47;
typedef cublasSideMode_t T44;
typedef cublasSideMode_t* T45;
typedef cublasSideMode_t& T46;
typedef cublasSideMode_t&& T47;

// CHECK: typedef oneapi::mkl::transpose T48;
// CHECK-NEXT: typedef oneapi::mkl::transpose* T49;
// CHECK-NEXT: typedef oneapi::mkl::transpose& T50;
// CHECK-NEXT: typedef oneapi::mkl::transpose&& T51;
typedef cublasOperation_t T48;
typedef cublasOperation_t* T49;
typedef cublasOperation_t& T50;
typedef cublasOperation_t&& T51;

// CHECK: typedef int T52;
// CHECK-NEXT: typedef int* T53;
// CHECK-NEXT: typedef int& T54;
// CHECK-NEXT: typedef int&& T55;
typedef cublasStatus T52;
typedef cublasStatus* T53;
typedef cublasStatus& T54;
typedef cublasStatus&& T55;

// CHECK: typedef int T56;
// CHECK-NEXT: typedef int* T57;
// CHECK-NEXT: typedef int& T58;
// CHECK-NEXT: typedef int&& T59;
typedef cusolverStatus_t T56;
typedef cusolverStatus_t* T57;
typedef cusolverStatus_t& T58;
typedef cusolverStatus_t&& T59;

// CHECK: typedef int64_t T60;
// CHECK-NEXT: typedef int64_t* T61;
// CHECK-NEXT: typedef int64_t& T62;
// CHECK-NEXT: typedef int64_t&& T63;
typedef cusolverEigType_t T60;
typedef cusolverEigType_t* T61;
typedef cusolverEigType_t& T62;
typedef cusolverEigType_t&& T63;

// CHECK: typedef oneapi::mkl::job T64;
// CHECK-NEXT: typedef oneapi::mkl::job* T65;
// CHECK-NEXT: typedef oneapi::mkl::job& T66;
// CHECK-NEXT: typedef oneapi::mkl::job&& T67;
typedef cusolverEigMode_t T64;
typedef cusolverEigMode_t* T65;
typedef cusolverEigMode_t& T66;
typedef cusolverEigMode_t&& T67;

// CHECK: typedef int T68;
// CHECK-NEXT: typedef int* T69;
// CHECK-NEXT: typedef int& T70;
// CHECK-NEXT: typedef int&& T71;
typedef curandStatus_t T68;
typedef curandStatus_t* T69;
typedef curandStatus_t& T70;
typedef curandStatus_t&& T71;

// CHECK: typedef int T72;
// CHECK-NEXT: typedef int* T73;
// CHECK-NEXT: typedef int& T74;
// CHECK-NEXT: typedef int&& T75;
typedef cufftResult_t T72;
typedef cufftResult_t* T73;
typedef cufftResult_t& T74;
typedef cufftResult_t&& T75;

// CHECK: typedef dpct::queue_ptr T76;
// CHECK-NEXT: typedef dpct::queue_ptr* T77;
// CHECK-NEXT: typedef dpct::queue_ptr& T78;
// CHECK-NEXT: typedef dpct::queue_ptr&& T79;
typedef cudaStream_t T76;
typedef cudaStream_t* T77;
typedef cudaStream_t& T78;
typedef cudaStream_t&& T79;

// CHECK: typedef sycl::queue T_1;
// CHECK-NEXT: typedef sycl::queue* T_2;
// CHECK-NEXT: typedef sycl::queue& T_3;
// CHECK-NEXT: typedef sycl::queue&& T_4;
typedef CUstream_st T_1;
typedef CUstream_st* T_2;
typedef CUstream_st& T_3;
typedef CUstream_st&& T_4;


// case 8
__device__ void foo_t(){

{
// CHECK: #define T8_0 sycl::range<3>
// CHECK-NEXT: #define T8_1 sycl::range<3> *
// CHECK-NEXT: #define T8_2 sycl::range<3> &
// CHECK-NEXT: #define T8_3 sycl::range<3> &&
// CHECK-NEXT:     T8_0 a1(1, 1, 1);
// CHECK-NEXT:     T8_1 a2;
// CHECK-NEXT:     T8_2 a3=a1;
// CHECK-NEXT:     T8_3 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_0 dim3
#define T8_1 dim3 *
#define T8_2 dim3 &
#define T8_3 dim3 &&
    T8_0 a1;
    T8_1 a2;
    T8_2 a3=a1;
    T8_3 a4=std::move(a1);
}

{
// CHECK: #define T8_4 int
// CHECK-NEXT: #define T8_5 int *
// CHECK-NEXT: #define T8_6 int &
// CHECK-NEXT: #define T8_7 int &&
// CHECK-NEXT:     T8_4 a1;
// CHECK-NEXT:     T8_5 a2;
// CHECK-NEXT:     T8_6 a3=a1;
// CHECK-NEXT:     T8_7 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_4 cudaError_t
#define T8_5 cudaError_t *
#define T8_6 cudaError_t &
#define T8_7 cudaError_t &&
    T8_4 a1;
    T8_5 a2;
    T8_6 a3=a1;
    T8_7 a4=std::move(a1);
}

{
// CHECK: #define T8_8 int
// CHECK-NEXT: #define T8_9 int *
// CHECK-NEXT: #define T8_10 int &
// CHECK-NEXT: #define T8_11 int &&
// CHECK-NEXT:     T8_8 a1;
// CHECK-NEXT:     T8_9 a2;
// CHECK-NEXT:     T8_10 a3=a1;
// CHECK-NEXT:     T8_11 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_8 cudaError
#define T8_9 cudaError *
#define T8_10 cudaError &
#define T8_11 cudaError &&
    T8_8 a1;
    T8_9 a2;
    T8_10 a3=a1;
    T8_11 a4=std::move(a1);
}

{
// CHECK: #define T8_12 int
// CHECK-NEXT: #define T8_13 int *
// CHECK-NEXT: #define T8_14 int &
// CHECK-NEXT: #define T8_15 int &&
// CHECK-NEXT:     T8_12 a1;
// CHECK-NEXT:     T8_13 a2;
// CHECK-NEXT:     T8_14 a3=a1;
// CHECK-NEXT:     T8_15 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_12 CUresult
#define T8_13 CUresult *
#define T8_14 CUresult &
#define T8_15 CUresult &&
    T8_12 a1;
    T8_13 a2;
    T8_14 a3=a1;
    T8_15 a4=std::move(a1);
}

{
// CHECK: #define T8_16 dpct::event_ptr
// CHECK-NEXT: #define T8_17 dpct::event_ptr *
// CHECK-NEXT: #define T8_18 dpct::event_ptr &
// CHECK-NEXT: #define T8_19 dpct::event_ptr &&
// CHECK-NEXT:     T8_16 a1;
// CHECK-NEXT:     T8_17 a2;
// CHECK-NEXT:     T8_18 a3=a1;
// CHECK-NEXT:     T8_19 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_16 cudaEvent_t
#define T8_17 cudaEvent_t *
#define T8_18 cudaEvent_t &
#define T8_19 cudaEvent_t &&
    T8_16 a1;
    T8_17 a2;
    T8_18 a3=a1;
    T8_19 a4=std::move(a1);
}

{
// CHECK: /*
// CHECK-NEXT: DPCT1021:{{[0-9]+}}: Migration of cublasHandle_t in __global__ or __device__ function is not supported. You may need to redesign the code.
// CHECK-NEXT: */
// CHECK-NEXT: #define T8_20 cublasHandle_t
// CHECK-NEXT: /*
// CHECK-NEXT: DPCT1021:{{[0-9]+}}: Migration of cublasHandle_t in __global__ or __device__ function is not supported. You may need to redesign the code.
// CHECK-NEXT: */
// CHECK-NEXT: #define T8_21 cublasHandle_t *
// CHECK-NEXT: /*
// CHECK-NEXT: DPCT1021:{{[0-9]+}}: Migration of cublasHandle_t in __global__ or __device__ function is not supported. You may need to redesign the code.
// CHECK-NEXT: */
// CHECK-NEXT: #define T8_22 cublasHandle_t &
// CHECK-NEXT: /*
// CHECK-NEXT: DPCT1021:{{[0-9]+}}: Migration of cublasHandle_t in __global__ or __device__ function is not supported. You may need to redesign the code.
// CHECK-NEXT: */
// CHECK-NEXT: #define T8_23 cublasHandle_t &&
// CHECK-NEXT:     T8_20 a1;
// CHECK-NEXT:     T8_21 a2;
// CHECK-NEXT:     T8_22 a3=a1;
// CHECK-NEXT:     T8_23 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_20 cublasHandle_t
#define T8_21 cublasHandle_t *
#define T8_22 cublasHandle_t &
#define T8_23 cublasHandle_t &&
    T8_20 a1;
    T8_21 a2;
    T8_22 a3=a1;
    T8_23 a4=std::move(a1);
}

{
// CHECK: #define T8_24 int
// CHECK-NEXT: #define T8_25 int *
// CHECK-NEXT: #define T8_26 int &
// CHECK-NEXT: #define T8_27 int &&
// CHECK-NEXT:     T8_24 a1;
// CHECK-NEXT:     T8_25 a2;
// CHECK-NEXT:     T8_26 a3=a1;
// CHECK-NEXT:     T8_27 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_24 cublasStatus_t
#define T8_25 cublasStatus_t *
#define T8_26 cublasStatus_t &
#define T8_27 cublasStatus_t &&
    T8_24 a1;
    T8_25 a2;
    T8_26 a3=a1;
    T8_27 a4=std::move(a1);
}

{
// CHECK: #define T8_28 sycl::float2
// CHECK-NEXT: #define T8_29 sycl::float2 *
// CHECK-NEXT: #define T8_30 sycl::float2 &
// CHECK-NEXT: #define T8_31 sycl::float2 &&
// CHECK-NEXT:     T8_28 a1;
// CHECK-NEXT:     T8_29 a2;
// CHECK-NEXT:     T8_30 a3=a1;
// CHECK-NEXT:     T8_31 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_28 cuComplex
#define T8_29 cuComplex *
#define T8_30 cuComplex &
#define T8_31 cuComplex &&
    T8_28 a1;
    T8_29 a2;
    T8_30 a3=a1;
    T8_31 a4=std::move(a1);
}

{
// CHECK: #define T8_32 sycl::double2
// CHECK-NEXT: #define T8_33 sycl::double2 *
// CHECK-NEXT: #define T8_34 sycl::double2 &
// CHECK-NEXT: #define T8_35 sycl::double2 &&
// CHECK-NEXT:     T8_32 a1;
// CHECK-NEXT:     T8_33 a2;
// CHECK-NEXT:     T8_34 a3=a1;
// CHECK-NEXT:     T8_35 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_32 cuDoubleComplex
#define T8_33 cuDoubleComplex *
#define T8_34 cuDoubleComplex &
#define T8_35 cuDoubleComplex &&
    T8_32 a1;
    T8_33 a2;
    T8_34 a3=a1;
    T8_35 a4=std::move(a1);
}

{
// CHECK: #define T8_36 oneapi::mkl::uplo
// CHECK-NEXT: #define T8_37 oneapi::mkl::uplo *
// CHECK-NEXT: #define T8_38 oneapi::mkl::uplo &
// CHECK-NEXT: #define T8_39 oneapi::mkl::uplo &&
// CHECK-NEXT:     T8_36 a1;
// CHECK-NEXT:     T8_37 a2;
// CHECK-NEXT:     T8_38 a3=a1;
// CHECK-NEXT:     T8_39 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_36 cublasFillMode_t
#define T8_37 cublasFillMode_t *
#define T8_38 cublasFillMode_t &
#define T8_39 cublasFillMode_t &&
    T8_36 a1;
    T8_37 a2;
    T8_38 a3=a1;
    T8_39 a4=std::move(a1);
}

{
// CHECK: #define T8_40 oneapi::mkl::diag
// CHECK-NEXT: #define T8_41 oneapi::mkl::diag *
// CHECK-NEXT: #define T8_42 oneapi::mkl::diag &
// CHECK-NEXT: #define T8_43 oneapi::mkl::diag &&
// CHECK-NEXT:     T8_40 a1;
// CHECK-NEXT:     T8_41 a2;
// CHECK-NEXT:     T8_42 a3=a1;
// CHECK-NEXT:     T8_43 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_40 cublasDiagType_t
#define T8_41 cublasDiagType_t *
#define T8_42 cublasDiagType_t &
#define T8_43 cublasDiagType_t &&
    T8_40 a1;
    T8_41 a2;
    T8_42 a3=a1;
    T8_43 a4=std::move(a1);
}

{
// CHECK: #define T8_44 oneapi::mkl::side
// CHECK-NEXT: #define T8_45 oneapi::mkl::side *
// CHECK-NEXT: #define T8_46 oneapi::mkl::side &
// CHECK-NEXT: #define T8_47 oneapi::mkl::side &&
// CHECK-NEXT:     T8_44 a1;
// CHECK-NEXT:     T8_45 a2;
// CHECK-NEXT:     T8_46 a3=a1;
// CHECK-NEXT:     T8_47 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_44 cublasSideMode_t
#define T8_45 cublasSideMode_t *
#define T8_46 cublasSideMode_t &
#define T8_47 cublasSideMode_t &&
    T8_44 a1;
    T8_45 a2;
    T8_46 a3=a1;
    T8_47 a4=std::move(a1);
}

{
// CHECK: #define T8_48 oneapi::mkl::transpose
// CHECK-NEXT: #define T8_49 oneapi::mkl::transpose *
// CHECK-NEXT: #define T8_50 oneapi::mkl::transpose &
// CHECK-NEXT: #define T8_51 oneapi::mkl::transpose &&
// CHECK-NEXT:     T8_48 a1;
// CHECK-NEXT:     T8_49 a2;
// CHECK-NEXT:     T8_50 a3=a1;
// CHECK-NEXT:     T8_51 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_48 cublasOperation_t
#define T8_49 cublasOperation_t *
#define T8_50 cublasOperation_t &
#define T8_51 cublasOperation_t &&
    T8_48 a1;
    T8_49 a2;
    T8_50 a3=a1;
    T8_51 a4=std::move(a1);
}

{
// CHECK: #define T8_52 int
// CHECK-NEXT: #define T8_53 int *
// CHECK-NEXT: #define T8_54 int &
// CHECK-NEXT: #define T8_55 int &&
// CHECK-NEXT:     T8_52 a1;
// CHECK-NEXT:     T8_53 a2;
// CHECK-NEXT:     T8_54 a3=a1;
// CHECK-NEXT:     T8_55 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_52 cublasStatus
#define T8_53 cublasStatus *
#define T8_54 cublasStatus &
#define T8_55 cublasStatus &&
    T8_52 a1;
    T8_53 a2;
    T8_54 a3=a1;
    T8_55 a4=std::move(a1);
}

{
// CHECK: #define T8_56 int
// CHECK-NEXT: #define T8_57 int *
// CHECK-NEXT: #define T8_58 int &
// CHECK-NEXT: #define T8_59 int &&
// CHECK-NEXT:     T8_56 a1;
// CHECK-NEXT:     T8_57 a2;
// CHECK-NEXT:     T8_58 a3=a1;
// CHECK-NEXT:     T8_59 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_56 cusolverStatus_t
#define T8_57 cusolverStatus_t *
#define T8_58 cusolverStatus_t &
#define T8_59 cusolverStatus_t &&
    T8_56 a1;
    T8_57 a2;
    T8_58 a3=a1;
    T8_59 a4=std::move(a1);
}

{
// CHECK: #define T8_60 int64_t
// CHECK-NEXT: #define T8_61 int64_t *
// CHECK-NEXT: #define T8_62 int64_t &
// CHECK-NEXT: #define T8_63 int64_t &&
// CHECK-NEXT:     T8_60 a1;
// CHECK-NEXT:     T8_61 a2;
// CHECK-NEXT:     T8_62 a3=a1;
// CHECK-NEXT:     T8_63 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_60 cusolverEigType_t
#define T8_61 cusolverEigType_t *
#define T8_62 cusolverEigType_t &
#define T8_63 cusolverEigType_t &&
    T8_60 a1;
    T8_61 a2;
    T8_62 a3=a1;
    T8_63 a4=std::move(a1);
}

{
// CHECK: #define T8_64 oneapi::mkl::job
// CHECK-NEXT: #define T8_65 oneapi::mkl::job *
// CHECK-NEXT: #define T8_66 oneapi::mkl::job &
// CHECK-NEXT: #define T8_67 oneapi::mkl::job &&
// CHECK-NEXT:     T8_64 a1;
// CHECK-NEXT:     T8_65 a2;
// CHECK-NEXT:     T8_66 a3=a1;
// CHECK-NEXT:     T8_67 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_64 cusolverEigMode_t
#define T8_65 cusolverEigMode_t *
#define T8_66 cusolverEigMode_t &
#define T8_67 cusolverEigMode_t &&
    T8_64 a1;
    T8_65 a2;
    T8_66 a3=a1;
    T8_67 a4=std::move(a1);
}

{
// CHECK: #define T8_68 int
// CHECK-NEXT: #define T8_69 int *
// CHECK-NEXT: #define T8_70 int &
// CHECK-NEXT: #define T8_71 int &&
// CHECK-NEXT:     T8_68 a1;
// CHECK-NEXT:     T8_69 a2;
// CHECK-NEXT:     T8_70 a3=a1;
// CHECK-NEXT:     T8_71 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_68 curandStatus_t
#define T8_69 curandStatus_t *
#define T8_70 curandStatus_t &
#define T8_71 curandStatus_t &&
    T8_68 a1;
    T8_69 a2;
    T8_70 a3=a1;
    T8_71 a4=std::move(a1);
}

{
// CHECK: #define T8_72 int
// CHECK-NEXT: #define T8_73 int *
// CHECK-NEXT: #define T8_74 int &
// CHECK-NEXT: #define T8_75 int &&
// CHECK-NEXT:     T8_72 a1;
// CHECK-NEXT:     T8_73 a2;
// CHECK-NEXT:     T8_74 a3=a1;
// CHECK-NEXT:     T8_75 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_72 cufftResult_t
#define T8_73 cufftResult_t *
#define T8_74 cufftResult_t &
#define T8_75 cufftResult_t &&
    T8_72 a1;
    T8_73 a2;
    T8_74 a3=a1;
    T8_75 a4=std::move(a1);
}

{
// CHECK: #define T8_76 dpct::queue_ptr
// CHECK-NEXT: #define T8_77 dpct::queue_ptr *
// CHECK-NEXT: #define T8_78 dpct::queue_ptr &
// CHECK-NEXT: #define T8_79 dpct::queue_ptr &&
// CHECK-NEXT:     T8_76 a1;
// CHECK-NEXT:     T8_77 a2;
// CHECK-NEXT:     T8_78 a3=a1;
// CHECK-NEXT:     T8_79 a4=std::move(a1);
// CHECK-NEXT: }
#define T8_76 cudaStream_t
#define T8_77 cudaStream_t *
#define T8_78 cudaStream_t &
#define T8_79 cudaStream_t &&
    T8_76 a1;
    T8_77 a2;
    T8_78 a3=a1;
    T8_79 a4=std::move(a1);
}

{
// CHECK: #define T8__2 sycl::queue *
// CHECK-NEXT: T8__2 a2;
#define T8__2 CUstream_st *
  T8__2 a2;
}
}


// case 9
template <typename T> void template_foo(T var) {}
#define foo1(DataType) template_foo(DataType varname)
#define foo2(DataType) template_foo(DataType * varname)
#define foo3(DataType) template_foo(DataType & varname)
#define foo4(DataType) template_foo(DataType && varname)

// CHECK: template <> void foo1(sycl::range<3>){}
// CHECK-NEXT: template <> void foo2(sycl::range<3>){}
// CHECK-NEXT: template <> void foo3(sycl::range<3>){}
// CHECK-NEXT: template <> void foo4(sycl::range<3>){}
template <> void foo1(dim3){}
template <> void foo2(dim3){}
template <> void foo3(dim3){}
template <> void foo4(dim3){}

// CHECK: template <> void foo1(int){}
// CHECK-NEXT: template <> void foo2(int){}
// CHECK-NEXT: template <> void foo3(int){}
// CHECK-NEXT: template <> void foo4(int){}
template <> void foo1(cudaError){}
template <> void foo2(cudaError){}
template <> void foo3(cudaError){}
template <> void foo4(cudaError){}

// CHECK: template <> void foo1(int){}
// CHECK-NEXT: template <> void foo2(int){}
// CHECK-NEXT: template <> void foo3(int){}
// CHECK-NEXT: template <> void foo4(int){}
template <> void foo1(CUresult){}
template <> void foo2(CUresult){}
template <> void foo3(CUresult){}
template <> void foo4(CUresult){}

// CHECK: template <> void foo1(dpct::event_ptr){}
// CHECK-NEXT: template <> void foo2(dpct::event_ptr){}
// CHECK-NEXT: template <> void foo3(dpct::event_ptr){}
// CHECK-NEXT: template <> void foo4(dpct::event_ptr){}
template <> void foo1(cudaEvent_t){}
template <> void foo2(cudaEvent_t){}
template <> void foo3(cudaEvent_t){}
template <> void foo4(cudaEvent_t){}

// CHECK: template <> void foo1(dpct::queue_ptr){}
// CHECK-NEXT: template <> void foo2(dpct::queue_ptr){}
// CHECK-NEXT: template <> void foo3(dpct::queue_ptr){}
// CHECK-NEXT: template <> void foo4(dpct::queue_ptr){}
template <> void foo1(cublasHandle_t){}
template <> void foo2(cublasHandle_t){}
template <> void foo3(cublasHandle_t){}
template <> void foo4(cublasHandle_t){}

// CHECK: template <> void foo1(sycl::float2){}
// CHECK-NEXT: template <> void foo2(sycl::float2){}
// CHECK-NEXT: template <> void foo3(sycl::float2){}
// CHECK-NEXT: template <> void foo4(sycl::float2){}
template <> void foo1(cuComplex){}
template <> void foo2(cuComplex){}
template <> void foo3(cuComplex){}
template <> void foo4(cuComplex){}

// CHECK: template <> void foo1(sycl::double2){}
// CHECK-NEXT: template <> void foo2(sycl::double2){}
// CHECK-NEXT: template <> void foo3(sycl::double2){}
// CHECK-NEXT: template <> void foo4(sycl::double2){}
template <> void foo1(cuDoubleComplex){}
template <> void foo2(cuDoubleComplex){}
template <> void foo3(cuDoubleComplex){}
template <> void foo4(cuDoubleComplex){}

// CHECK: template <> void foo1(oneapi::mkl::uplo){}
// CHECK-NEXT: template <> void foo2(oneapi::mkl::uplo){}
// CHECK-NEXT: template <> void foo3(oneapi::mkl::uplo){}
// CHECK-NEXT: template <> void foo4(oneapi::mkl::uplo){}
template <> void foo1(cublasFillMode_t){}
template <> void foo2(cublasFillMode_t){}
template <> void foo3(cublasFillMode_t){}
template <> void foo4(cublasFillMode_t){}

// CHECK: template <> void foo1(oneapi::mkl::diag){}
// CHECK-NEXT: template <> void foo2(oneapi::mkl::diag){}
// CHECK-NEXT: template <> void foo3(oneapi::mkl::diag){}
// CHECK-NEXT: template <> void foo4(oneapi::mkl::diag){}
template <> void foo1(cublasDiagType_t){}
template <> void foo2(cublasDiagType_t){}
template <> void foo3(cublasDiagType_t){}
template <> void foo4(cublasDiagType_t){}

// CHECK: template <> void foo1(oneapi::mkl::side){}
// CHECK-NEXT: template <> void foo2(oneapi::mkl::side){}
// CHECK-NEXT: template <> void foo3(oneapi::mkl::side){}
// CHECK-NEXT: template <> void foo4(oneapi::mkl::side){}
template <> void foo1(cublasSideMode_t){}
template <> void foo2(cublasSideMode_t){}
template <> void foo3(cublasSideMode_t){}
template <> void foo4(cublasSideMode_t){}

// CHECK: template <> void foo1(oneapi::mkl::transpose){}
// CHECK-NEXT: template <> void foo2(oneapi::mkl::transpose){}
// CHECK-NEXT: template <> void foo3(oneapi::mkl::transpose){}
// CHECK-NEXT: template <> void foo4(oneapi::mkl::transpose){}
template <> void foo1(cublasOperation_t){}
template <> void foo2(cublasOperation_t){}
template <> void foo3(cublasOperation_t){}
template <> void foo4(cublasOperation_t){}

// CHECK: template <> void foo1(int){}
// CHECK-NEXT: template <> void foo2(int){}
// CHECK-NEXT: template <> void foo3(int){}
// CHECK-NEXT: template <> void foo4(int){}
template <> void foo1(cublasStatus_t){}
template <> void foo2(cublasStatus_t){}
template <> void foo3(cublasStatus_t){}
template <> void foo4(cublasStatus_t){}

// CHECK: template <> void foo1(int){}
// CHECK-NEXT: template <> void foo2(int){}
// CHECK-NEXT: template <> void foo3(int){}
// CHECK-NEXT: template <> void foo4(int){}
template <> void foo1(cusolverStatus_t){}
template <> void foo2(cusolverStatus_t){}
template <> void foo3(cusolverStatus_t){}
template <> void foo4(cusolverStatus_t){}

// CHECK: template <> void foo1(int64_t){}
// CHECK-NEXT: template <> void foo2(int64_t){}
// CHECK-NEXT: template <> void foo3(int64_t){}
// CHECK-NEXT: template <> void foo4(int64_t){}
template <> void foo1(cusolverEigType_t){}
template <> void foo2(cusolverEigType_t){}
template <> void foo3(cusolverEigType_t){}
template <> void foo4(cusolverEigType_t){}

// CHECK: template <> void foo1(oneapi::mkl::job){}
// CHECK-NEXT: template <> void foo2(oneapi::mkl::job){}
// CHECK-NEXT: template <> void foo3(oneapi::mkl::job){}
// CHECK-NEXT: template <> void foo4(oneapi::mkl::job){}
template <> void foo1(cusolverEigMode_t){}
template <> void foo2(cusolverEigMode_t){}
template <> void foo3(cusolverEigMode_t){}
template <> void foo4(cusolverEigMode_t){}

// CHECK: template <> void foo1(int){}
// CHECK-NEXT: template <> void foo2(int){}
// CHECK-NEXT: template <> void foo3(int){}
// CHECK-NEXT: template <> void foo4(int){}
template <> void foo1(curandStatus_t){}
template <> void foo2(curandStatus_t){}
template <> void foo3(curandStatus_t){}
template <> void foo4(curandStatus_t){}

// CHECK: template <> void foo1(int){}
// CHECK-NEXT: template <> void foo2(int){}
// CHECK-NEXT: template <> void foo3(int){}
// CHECK-NEXT: template <> void foo4(int){}
template <> void foo1(cufftResult_t){}
template <> void foo2(cufftResult_t){}
template <> void foo3(cufftResult_t){}
template <> void foo4(cufftResult_t){}

// CHECK: template <> void foo1(dpct::queue_ptr){}
// CHECK-NEXT: template <> void foo2(dpct::queue_ptr){}
// CHECK-NEXT: template <> void foo3(dpct::queue_ptr){}
// CHECK-NEXT: template <> void foo4(dpct::queue_ptr){}
template <> void foo1(cudaStream_t){}
template <> void foo2(cudaStream_t){}
template <> void foo3(cudaStream_t){}
template <> void foo4(cudaStream_t){}

void foo_struct(void) {
// CHECK: dpct::device_info d_t;
struct cudaDeviceProp d_t;
}

// CHECK: void foo(dpct::queue_ptr& stream) {
// CHECK-NEXT:   dpct::queue_ptr s0;
// CHECK-NEXT:   dpct::queue_ptr &s1 = s0;
// CHECK-NEXT: }
void foo(cudaStream_t& stream) {
  cudaStream_t s0;
  cudaStream_t &s1 = s0;
}
