// RUN: cd %T
// RUN: cat %S/../../nccl_test.h > %T/../nccl_test.h
// RUN: cat %S/../../cudnn_test.h > %T/../cudnn_test.h
// RUN: cat %s > %T/api_is_inroot.cu
// RUN: dpct --format-range=none ./api_is_inroot.cu --in-root=.. --out-root=%T/out --cuda-include-path="%cuda-path/include" -- --cuda-host-only -I..
// RUN: FileCheck %s --match-full-lines --input-file %T/out/Output/api_is_inroot.dp.cpp
// RUN: cd ..
// RUN: rm -rf ./*


//cudnn_test.h and nccl_test.h are both in inroot, so do not emit warnings.

//CHECK:#include <CL/sycl.hpp>
//CHECK-NEXT:#include <dpct/dpct.hpp>
//CHECK-NEXT:#include <cstdio>
//CHECK-NEXT:#include <cudnn_test.h>
//CHECK-NEXT:#include <nccl_test.h>
#include <cstdio>
#include <cudnn_test.h>
#include <nccl_test.h>
#include <cuda_runtime.h>

int main() {
//CHECK:int a1 = cudnnAAA();
//CHECK-NEXT:cudnnStatus_t b1;
//CHECK-NEXT:cudnnStatus c1;
//CHECK-NEXT:if(b1 == CUDNN_SUCCESS){
//CHECK-NEXT:}
//CHECK-NEXT:cudnnCLASS d1;
//CHECK-NEXT:cudnnTemplateCLASS<double> e1;
//CHECK-NEXT:ncclAAA();
//CHECK-NEXT:ncclStatus_t b2;
//CHECK-NEXT:ncclStatus c2;
//CHECK-NEXT:if(b2 == NCCL_SUCCESS){
//CHECK-NEXT:}
//CHECK-NEXT:ncclCLASS d2;
//CHECK-NEXT:ncclTemplateCLASS<float> e2;

int a1 = cudnnAAA();
cudnnStatus_t b1;
cudnnStatus c1;
if(b1 == CUDNN_SUCCESS){
}
cudnnCLASS d1;
cudnnTemplateCLASS<double> e1;
ncclAAA();
ncclStatus_t b2;
ncclStatus c2;
if(b2 == NCCL_SUCCESS){
}
ncclCLASS d2;
ncclTemplateCLASS<float> e2;
}
