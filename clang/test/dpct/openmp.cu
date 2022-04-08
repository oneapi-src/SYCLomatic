// UNSUPPORTED: -windows-
// RUN: c2s --format-range=none -out-root %T/openmp %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -fopenmp
// RUN: FileCheck %s --match-full-lines --input-file %T/openmp/openmp.dp.cpp

#include "stdio.h"
#include "omp.h"
__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

int main() {
    cuda_hello<<<1,1>>>(); 
    cudaDeviceSynchronize();
    return 0;
}


//CHECK: int foo(){
//CHECK-NEXT:     int i = 0;
//CHECK-NEXT:     int b = 0;
//CHECK-NEXT: #pragma omp parallel
//CHECK-NEXT:     for(i=0;i<10;i++){
//CHECK-NEXT:         b = b + 2;
//CHECK-NEXT:     }
//CHECK-NEXT:     return 0;
//CHECK-NEXT: }
int foo(){
    int i = 0;
    int b = 0;
#pragma omp parallel
    for(i=0;i<10;i++){
        b = b + 2;
    }
    return 0;
}
