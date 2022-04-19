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
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1037:{{[0-9]+}}: Rewrite this code using Intel(R) oneAPI Deep Neural Network Library (oneDNN) with DPC++.
//CHECK-NEXT:*/
//CHECK-NEXT:#include <cudnn_test.h>
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1037:{{[0-9]+}}: Rewrite this code using Intel(R) oneAPI Collective Communications Library with DPC++.
//CHECK-NEXT:*/
//CHECK-NEXT:#include <nccl_test.h>
#include <cstdio>
#include <cudnn_test.h>
#include <nccl_test.h>
#include <cuda_runtime.h>

int main() {
//CHECK:/*
//CHECK-NEXT:DPCT1037:{{[0-9]+}}: Rewrite this code using Intel(R) oneAPI Deep Neural Network Library (oneDNN) with DPC++.
//CHECK-NEXT:*/
//CHECK-NEXT:int a1 = cudnnAAA();
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1037:{{[0-9]+}}: Rewrite this code using Intel(R) oneAPI Deep Neural Network Library (oneDNN) with DPC++.
//CHECK-NEXT:*/
//CHECK-NEXT:cudnnFooType b1;
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1037:{{[0-9]+}}: Rewrite this code using Intel(R) oneAPI Deep Neural Network Library (oneDNN) with DPC++.
//CHECK-NEXT:*/
//CHECK-NEXT:cudnnFooEnum c1;
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1037:{{[0-9]+}}: Rewrite this code using Intel(R) oneAPI Deep Neural Network Library (oneDNN) with DPC++.
//CHECK-NEXT:*/
//CHECK-NEXT:if(b1 == CUDNN_FOO_VAL){
//CHECK-NEXT:}
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1037:{{[0-9]+}}: Rewrite this code using Intel(R) oneAPI Deep Neural Network Library (oneDNN) with DPC++.
//CHECK-NEXT:*/
//CHECK-NEXT:cudnnCLASS d1;
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1037:{{[0-9]+}}: Rewrite this code using Intel(R) oneAPI Deep Neural Network Library (oneDNN) with DPC++.
//CHECK-NEXT:*/
//CHECK-NEXT:cudnnTemplateCLASS<double> e1;
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1037:{{[0-9]+}}: Rewrite this code using Intel(R) oneAPI Collective Communications Library with DPC++.
//CHECK-NEXT:*/
//CHECK-NEXT:ncclAAA();
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1037:{{[0-9]+}}: Rewrite this code using Intel(R) oneAPI Collective Communications Library with DPC++.
//CHECK-NEXT:*/
//CHECK-NEXT:ncclFooType b2;
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1037:{{[0-9]+}}: Rewrite this code using Intel(R) oneAPI Collective Communications Library with DPC++.
//CHECK-NEXT:*/
//CHECK-NEXT:ncclFooEnum c2;
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1037:{{[0-9]+}}: Rewrite this code using Intel(R) oneAPI Collective Communications Library with DPC++.
//CHECK-NEXT:*/
//CHECK-NEXT:if(b2 == NCCL_FOO_VAL){
//CHECK-NEXT:}
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1037:{{[0-9]+}}: Rewrite this code using Intel(R) oneAPI Collective Communications Library with DPC++.
//CHECK-NEXT:*/
//CHECK-NEXT:ncclCLASS d2;
//CHECK-NEXT:/*
//CHECK-NEXT:DPCT1037:{{[0-9]+}}: Rewrite this code using Intel(R) oneAPI Collective Communications Library with DPC++.
//CHECK-NEXT:*/
//CHECK-NEXT:ncclTemplateCLASS<float> e2;

int a1 = cudnnAAA();
cudnnFooType b1;
cudnnFooEnum c1;
if(b1 == CUDNN_FOO_VAL){
}
cudnnCLASS d1;
cudnnTemplateCLASS<double> e1;
ncclAAA();
ncclFooType b2;
ncclFooEnum c2;
if(b2 == NCCL_FOO_VAL){
}
ncclCLASS d2;
ncclTemplateCLASS<float> e2;
}
