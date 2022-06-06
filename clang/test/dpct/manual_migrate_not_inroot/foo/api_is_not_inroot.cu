// RUN: cd %T
// RUN: cat %S/../../nccl_test.h > %T/../nccl_test.h
// RUN: cat %S/../../cudnn_test.h > %T/../cudnn_test.h
// RUN: cat %s > %T/api_is_not_inroot.cu
// RUN: dpct --format-range=none ./api_is_not_inroot.cu --in-root=. --out-root=%T/out --cuda-include-path="%cuda-path/include" -- --cuda-host-only -I..
// RUN: FileCheck %s --match-full-lines --input-file %T/out/api_is_not_inroot.dp.cpp
// RUN: cd ..
// RUN: rm -rf ./*


//cudnn_test.h and nccl_test.h are not in inroot, so emit warnings.

//CHECK:#include <CL/sycl.hpp>
//CHECK-NEXT:#include <dpct/dpct.hpp>
//CHECK-NEXT:#include <cstdio>
//CHECK-NEXT:#include <cudnn_test.h>
#include <cstdio>
#include <cudnn_test.h>
#include <cuda_runtime.h>

int main() {

int a1 = cudnnAAA();
cudnnFooType b1;
cudnnFooEnum c1;
if(b1 == CUDNN_FOO_VAL){
}
cudnnCLASS d1;
cudnnTemplateCLASS<double> e1;
}
