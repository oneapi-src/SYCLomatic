// RUN: dpct --format-range=none -out-root %T/cudaMallocHost-migration-optimization %s --optimize-migration --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/cudaMallocHost-migration-optimization/cudaMallocHost-migration-optimization.dp.cpp
#include <cuda_runtime.h>
#include <stdio.h>

void foo(float *p){
  int i = p[0];
};
void bar(float p){
    int i = p;
};

template<typename T>
double process(size_t index, T *z) {
  double result = 0.f;
  result = pow(z[index], 2);
  return sqrt(result);
}

void test1(){
    float *p;
// CHECK: p = (float *)malloc(10);
    cudaMallocHost(&p, 10);
// CHECK: free(p);
    cudaFreeHost(p);
}

void test2(){
    float *p;
// CHECK: p = (float *)malloc(10);
    cudaMallocHost(&p, 10);
    foo(p);
// CHECK: free(p);
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
    bar(*p);
    bar(p[0]);
// CHECK: free(p);
    cudaFreeHost(p);
}

void test5(){
    float *p;
    // CHECK: p = (float *)malloc(10);
    cudaMallocHost(&p, 10);
    process<double>(1, (double *)p);
    process<float>(1, p);
    // CHECK: free(p);
    cudaFreeHost(p);
}

void test6(){
    float *p;
    // CHECK: p = (float *)malloc(10);
    cudaMallocHost(&p, 10);
    printf("%d\n", p);
    // CHECK: free(p);
    cudaFreeHost(p);
}

void test7(){
    float *p;
    // CHECK: p = (float *)malloc(10);
    cudaMallocHost((void **)&p, 10);
    printf("%d\n", process<float>(1, p));
    // CHECK: free(p);
    cudaFreeHost(p);
}

void test8(){
    float *p, *q;
    // CHECK: p = (float *)malloc(10);
    cudaMallocHost((void **)&p, 10);
    cudaMalloc(&q, 10);
    memset(p, 0, 10);
    cudaMemcpyAsync(q, p, 10, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(p, q, 10, cudaMemcpyDeviceToHost);
    // CHECK: free(p);
    cudaFreeHost(p);
}

void test9(){
    float *p, *q;
    // CHECK: p = (float *)sycl::malloc_host(10, q_ct1);
    cudaMallocHost((void **)&p, 10);
    cudaMalloc(&q, 10);
    memset(p, 0, 10);
    cudaMemcpyAsync(q, p, 10, cudaMemcpyDeviceToDevice);
    // CHECK: sycl::free(p, q_ct1);
    cudaFreeHost(p);
}

void test10(){
    float *p;
    // CHECK: p = (float *)malloc(10);
    cudaMallocHost(&p, 10);
    printf("%d\n", p);
    // CHECK: free((void *)p);
    cudaFreeHost((void *)p);
}

int main(){
  test1();
  test2();
  test3();
  test4();
  test5();
  test6();
  test7();
  test8();
  test9();
  test10();
  return 0;
}