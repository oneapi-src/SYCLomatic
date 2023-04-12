// RUN: dpct --format-range=none -out-root %T/cudaMallocHost-migration-optimization %s --optimize-migration --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/cudaMallocHost-migration-optimization/cudaMallocHost-migration-optimization.dp.cpp
#include <cuda_runtime.h>
#include <stdio.h>

void a(float *p){
  int b = p[0];
};
void b(float p){
    int b = p;
};

template<typename T>
double rmsval(size_t size, T *z) {
  double rms = 0.0;
  int i;
  for (i = 0; i < size; i++) {
    rms = rms + pow(z[i], 2);
  }
  return sqrt(rms);
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
    a(p);
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
    b(*p);
    b(p[0]);
// CHECK: free(p);
    cudaFreeHost(p);
}

void test5(){
    float *p;
    // CHECK: p = (float *)malloc(10);
    cudaMallocHost(&p, 10);
    rmsval<double>(1, (double *)p);
    rmsval<float>(1, p);
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
    printf("%d\n", rmsval<float>(1, p));
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

int main(){
  test1();
  test2();
  test3();
  test4();
  test5();
  test6();
  test7();
  test8();
  return 0;
}