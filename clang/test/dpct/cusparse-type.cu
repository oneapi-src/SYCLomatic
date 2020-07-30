// RUN: dpct --format-range=none --out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cusparse-type.dp.cpp --match-full-lines %s
#include <cstdio>
#include <cusparse_v2.h>
#include <cuda_runtime.h>

int main(){
  //CHECK: oneapi::mkl::uplo a1;
  //CHECK-NEXT: a1 = oneapi::mkl::uplo::lower;
  //CHECK-NEXT: a1 = oneapi::mkl::uplo::upper;
  cusparseFillMode_t a1;
  a1 = CUSPARSE_FILL_MODE_LOWER;
  a1 = CUSPARSE_FILL_MODE_UPPER;

  //CHECK: oneapi::mkl::diag a2;
  //CHECK-NEXT: a2 = oneapi::mkl::diag::nonunit;
  //CHECK-NEXT: a2 = oneapi::mkl::diag::unit;
  cusparseDiagType_t a2;
  a2 = CUSPARSE_DIAG_TYPE_NON_UNIT;
  a2 = CUSPARSE_DIAG_TYPE_UNIT;

  //CHECK: oneapi::mkl::index_base a3;
  //CHECK-NEXT: a3 = oneapi::mkl::index_base::zero;
  //CHECK-NEXT: a3 = oneapi::mkl::index_base::one;
  cusparseIndexBase_t a3;
  a3 = CUSPARSE_INDEX_BASE_ZERO;
  a3 = CUSPARSE_INDEX_BASE_ONE;

  //CHECK: int a4;
  //CHECK-NEXT: a4 = 0;
  //CHECK-NEXT: a4 = 1;
  //CHECK-NEXT: a4 = 2;
  //CHECK-NEXT: a4 = 3;
  cusparseMatrixType_t a4;
  a4 = CUSPARSE_MATRIX_TYPE_GENERAL;
  a4 = CUSPARSE_MATRIX_TYPE_SYMMETRIC;
  a4 = CUSPARSE_MATRIX_TYPE_HERMITIAN;
  a4 = CUSPARSE_MATRIX_TYPE_TRIANGULAR;

  //CHECK: oneapi::mkl::transpose a5;
  //CHECK-NEXT: a5 = oneapi::mkl::transpose::nontrans;
  //CHECK-NEXT: a5 = oneapi::mkl::transpose::trans;
  //CHECK-NEXT: a5 = oneapi::mkl::transpose::conjtrans;
  cusparseOperation_t a5;
  a5 = CUSPARSE_OPERATION_NON_TRANSPOSE;
  a5 = CUSPARSE_OPERATION_TRANSPOSE;
  a5 = CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;

  //CHECK: int a6;
  //CHECK-NEXT: a6 = 0;
  //CHECK-NEXT: a6 = 1;
  //CHECK-NEXT: a6 = 2;
  //CHECK-NEXT: a6 = 3;
  //CHECK-NEXT: a6 = 4;
  //CHECK-NEXT: a6 = 5;
  //CHECK-NEXT: a6 = 6;
  //CHECK-NEXT: a6 = 7;
  //CHECK-NEXT: a6 = 8;
  //CHECK-NEXT: a6 = 9;
  cusparseStatus_t a6;
  a6 = CUSPARSE_STATUS_SUCCESS;
  a6 = CUSPARSE_STATUS_NOT_INITIALIZED;
  a6 = CUSPARSE_STATUS_ALLOC_FAILED;
  a6 = CUSPARSE_STATUS_INVALID_VALUE;
  a6 = CUSPARSE_STATUS_ARCH_MISMATCH;
  a6 = CUSPARSE_STATUS_MAPPING_ERROR;
  a6 = CUSPARSE_STATUS_EXECUTION_FAILED;
  a6 = CUSPARSE_STATUS_INTERNAL_ERROR;
  a6 = CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
  a6 = CUSPARSE_STATUS_ZERO_PIVOT;

  //CHECK: oneapi::mkl::index_base a7;
  cusparseMatDescr_t a7;

  //CHECK: sycl::queue* a8;
  cusparseHandle_t a8;
}

//CHECK: void foo(oneapi::mkl::uplo a1,
//CHECK-NEXT:     oneapi::mkl::diag a2,
//CHECK-NEXT:     oneapi::mkl::index_base a3,
//CHECK-NEXT:     int a4,
//CHECK-NEXT:     oneapi::mkl::transpose a5,
//CHECK-NEXT:     int a6,
//CHECK-NEXT:     oneapi::mkl::index_base a7,
//CHECK-NEXT:     sycl::queue* a8);
void foo(cusparseFillMode_t a1,
         cusparseDiagType_t a2,
         cusparseIndexBase_t a3,
         cusparseMatrixType_t a4,
         cusparseOperation_t a5,
         cusparseStatus_t a6,
         cusparseMatDescr_t a7,
         cusparseHandle_t a8);

//CHECK:oneapi::mkl::uplo foo1();
//CHECK-NEXT:oneapi::mkl::diag foo2();
//CHECK-NEXT:oneapi::mkl::index_base foo3();
//CHECK-NEXT:int foo4();
//CHECK-NEXT:oneapi::mkl::transpose foo5();
//CHECK-NEXT:int foo6();
//CHECK-NEXT:oneapi::mkl::index_base foo7();
//CHECK-NEXT:sycl::queue* foo8();
cusparseFillMode_t foo1();
cusparseDiagType_t foo2();
cusparseIndexBase_t foo3();
cusparseMatrixType_t foo4();
cusparseOperation_t foo5();
cusparseStatus_t foo6();
cusparseMatDescr_t foo7();
cusparseHandle_t foo8();

//CHECK:template<typename T>
//CHECK-NEXT:void bar1(oneapi::mkl::uplo a1,
//CHECK-NEXT:          oneapi::mkl::diag a2,
//CHECK-NEXT:          oneapi::mkl::index_base a3,
//CHECK-NEXT:          int a4,
//CHECK-NEXT:          oneapi::mkl::transpose a5,
//CHECK-NEXT:          int a6,
//CHECK-NEXT:          oneapi::mkl::index_base a7,
//CHECK-NEXT:          sycl::queue* a8){}
template<typename T>
void bar1(cusparseFillMode_t a1,
         cusparseDiagType_t a2,
         cusparseIndexBase_t a3,
         cusparseMatrixType_t a4,
         cusparseOperation_t a5,
         cusparseStatus_t a6,
         cusparseMatDescr_t a7,
         cusparseHandle_t a8){}

//CHECK:template<typename T>
//CHECK-NEXT:void bar2(oneapi::mkl::uplo a1,
//CHECK-NEXT:          oneapi::mkl::diag a2,
//CHECK-NEXT:          oneapi::mkl::index_base a3,
//CHECK-NEXT:          int a4,
//CHECK-NEXT:          oneapi::mkl::transpose a5,
//CHECK-NEXT:          int a6,
//CHECK-NEXT:          oneapi::mkl::index_base a7,
//CHECK-NEXT:          sycl::queue* a8){}
template<typename T>
void bar2(cusparseFillMode_t a1,
         cusparseDiagType_t a2,
         cusparseIndexBase_t a3,
         cusparseMatrixType_t a4,
         cusparseOperation_t a5,
         cusparseStatus_t a6,
         cusparseMatDescr_t a7,
         cusparseHandle_t a8){}

// specialization
//CHECK:template<>
//CHECK-NEXT:void bar2<double>(oneapi::mkl::uplo a1,
//CHECK-NEXT:             oneapi::mkl::diag a2,
//CHECK-NEXT:             oneapi::mkl::index_base a3,
//CHECK-NEXT:             int a4,
//CHECK-NEXT:             oneapi::mkl::transpose a5,
//CHECK-NEXT:             int a6,
//CHECK-NEXT:             oneapi::mkl::index_base a7,
//CHECK-NEXT:             sycl::queue* a8){}
template<>
void bar2<double>(cusparseFillMode_t a1,
                  cusparseDiagType_t a2,
                  cusparseIndexBase_t a3,
                  cusparseMatrixType_t a4,
                  cusparseOperation_t a5,
                  cusparseStatus_t a6,
                  cusparseMatDescr_t a7,
                  cusparseHandle_t a8){}


//explicit instantiation
//TODO: Since the parameter location of the explicit instantiation is same as the parameter in the primary template,
//tool currently cannot cover this case.
template void bar2<int>(cusparseFillMode_t a1,
                  cusparseDiagType_t a2,
                  cusparseIndexBase_t a3,
                  cusparseMatrixType_t a4,
                  cusparseOperation_t a5,
                  cusparseStatus_t a6,
                  cusparseMatDescr_t a7,
                  cusparseHandle_t a8);

//TODO: If the variables are non-local variables, the AST is like:
//TranslationUnitDecl 0x55bbd2993ba8 <<invalid sloc>> <invalid sloc>
//|-VarDecl 0x55888acd49e0 <test4.cu:5:1, col:24> col:20 b 'cusparseMatDescr_t':'struct cusparseMatDescr *' cinit
//| `-ImplicitCastExpr 0x55888acd4a60 <col:24> 'cusparseMatDescr_t':'struct cusparseMatDescr *' <NullToPointer>
//|   `-IntegerLiteral 0x55888acd4a40 <col:24> 'int' 0
//`-VarDecl 0x55888acd4a88 <col:1, col:31> col:27 c 'cusparseMatDescr_t':'struct cusparseMatDescr *' cinit
//  `-ImplicitCastExpr 0x55888acd4b08 <col:31> 'cusparseMatDescr_t':'struct cusparseMatDescr *' <NullToPointer>
//    `-IntegerLiteral 0x55888acd4ae8 <col:31> 'int' 0
//There is no DeclStmt node as these two VarDecl nodes' parent.
//Currently, tool cannot cover this case.
cusparseMatDescr_t b = 0, c = 0;