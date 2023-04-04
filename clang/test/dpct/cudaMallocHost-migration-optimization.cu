// RUN: dpct --format-range=none -out-root %T/cudaMallocHost-migration-optimization %s --optimize-migration --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/cudaMallocHost-migration-optimization/cudaMallocHost-migration-optimization.dp.cpp
#include <cuda_runtime.h>
#include <stdio.h>

void a(float *p){};
void b(float p){};
void test1(){
    float *p;
// CHECK: p = (float *)malloc(10);
    cudaMallocHost(&p, 10);
// CHECK: free(p);
    cudaFreeHost(p);
}

void test2(){
    float *p;
// CHECK: p = (float *)sycl::malloc_host(10, q_ct1);
    cudaMallocHost(&p, 10);
    a(p);
// CHECK: sycl::free(p, q_ct1);
    cudaFreeHost(p);
}

void test3(){
    float *p;
// CHECK: p = (float *)malloc(10);
    cudaMallocHost(&p, 10);
    printf("%f", p);
// CHECK: free(p);
    cudaFreeHost(p);
}

void test4(){
    float *p;
// CHECK: p = (float *)malloc(10);
    cudaMallocHost(&p, 10);
    b(*p);
    b(p[0]);
// CHECK: free(p);
    cudaFreeHost(p);
}

int main(){
  test1();
  test2();
  test3();
  return 0;
}