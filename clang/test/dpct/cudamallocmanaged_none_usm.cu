// RUN: dpct --format-range=none -usm-level=none -out-root %T/cudamallocmanaged_none_usm %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++14
// RUN: FileCheck %s --match-full-lines --input-file %T/cudamallocmanaged_none_usm/cudamallocmanaged_none_usm.dp.cpp

#define VECTOR_SIZE 256
#include<cuda_runtime.h>
#include<stdio.h>

__global__ void ker(int *A, int *B, int *C){
  C[threadIdx.x] = A[threadIdx.x] + B[threadIdx.x];
}

struct A{
  struct B{
    struct C{
      int *a;
    };
    C c;
  };
  B b;
};

template<typename T>
void temp1(){
  T* a;
  T b = 0;

  // CHECK: a = (T *)dpct::dpct_malloc(10 * sizeof(T));
  cudaMallocManaged(&a, 10 * sizeof(T));

  // CHECK: b = dpct::get_host_ptr<T>(a)[2];
  // CHECK-NEXT: b = *dpct::get_host_ptr<T>(a);
  // CHECK-NEXT: b = *dpct::get_host_ptr<T>(a++);
  // CHECK-NEXT: b = *dpct::get_host_ptr<T>(++a);
  // CHECK-NEXT: b = *dpct::get_host_ptr<T>(a--);
  // CHECK-NEXT: b = *dpct::get_host_ptr<T>(--a);
  // CHECK-NEXT: b = *dpct::get_host_ptr<T>(a + 2);
  // CHECK-NEXT: b = *dpct::get_host_ptr<T>(a) + 1;
  b = a[2];
  b = *a;
  b = *a++;
  b = *++a;
  b = *a--;
  b = *--a;
  b = *(a + 2);
  b = *a + 1;

  // CHECK: dpct::get_host_ptr<T>(a)[2] = b;
  // CHECK-NEXT: *dpct::get_host_ptr<T>(a) = b;
  // CHECK-NEXT: *dpct::get_host_ptr<T>(a++) = b;
  // CHECK-NEXT: *dpct::get_host_ptr<T>(++a) = b;
  // CHECK-NEXT: *dpct::get_host_ptr<T>(a--) = b;
  // CHECK-NEXT: *dpct::get_host_ptr<T>(a + 2) = b;
  a[2] = b;
  *a = b;
  *a++ = b;
  *++a = b;
  *a-- = b;
  *(a + 2) = b;

  cudaFree(a);
}

template<typename T>
void temp2(){
  T* a, *b, *c;
  // CHECK: /*
  // CHECK-NEXT: DPCT1070:{{[0-9]+}}: 'a' is allocated by dpct::dpct_malloc. Use 'dpct::get_host_ptr<T>(a)' to access the pointer from the host code.
  // CHECK-NEXT: */
  // CHECK-NEXT: a = (T *)dpct::dpct_malloc(10 * sizeof(T));
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1070:{{[0-9]+}}: 'b' is allocated by dpct::dpct_malloc. Use 'dpct::get_host_ptr<T>(b)' to access the pointer from the host code.
  // CHECK-NEXT: */
  // CHECK-NEXT: b = (T *)dpct::dpct_malloc(10 * sizeof(T));
  // CHECK-NEXT: a = (T *)dpct::dpct_malloc(10 * sizeof(T));
  cudaMallocManaged(&a, 10 * sizeof(T));
  cudaMallocManaged(&b, 10 * sizeof(T));
  cudaMalloc(&a, 10 * sizeof(T));

  c = b;
  // CHECK: a[0] = 1;
  a[0] = 1;
  // CHECK: dpct::get_host_ptr<T>(b)[0] = 1;
  b[0] = 1;

  cudaFree(a);
  cudaFree(b);
}

template<typename T>
void temp3(){
  A aa;
  // CHECK: /*
  // CHECK-NEXT: DPCT1070:{{[0-9]+}}: 'aa.b.c.a' is allocated by dpct::dpct_malloc. Use 'dpct::get_host_ptr<int>(aa.b.c.a)' to access the pointer from the host code.
  // CHECK-NEXT: */
  // CHECK-NEXT: aa.b.c.a = (int *)dpct::dpct_malloc(10 * sizeof(int));
  cudaMallocManaged(&aa.b.c.a, 10 * sizeof(int));

  // CHECK: aa.b.c.a[0] = 1;
  aa.b.c.a[0] = 1;

  cudaFree(aa.b.c.a);
}

class A1{
  int* a;
public:
  A1(){
    // CHECK: a = (int *)dpct::dpct_malloc(10 * sizeof(int));
    cudaMallocManaged(&a, 10 * sizeof(int));
  };
  void run1(){
    int b;
    // CHECK: b = dpct::get_host_ptr<int>(a)[2];
    // CHECK-NEXT: b = *dpct::get_host_ptr<int>(a);
    // CHECK-NEXT: b = *dpct::get_host_ptr<int>(a++);
    // CHECK-NEXT: b = *dpct::get_host_ptr<int>(++a);
    // CHECK-NEXT: b = *dpct::get_host_ptr<int>(a--);
    // CHECK-NEXT: b = *dpct::get_host_ptr<int>(--a);
    // CHECK-NEXT: b = *dpct::get_host_ptr<int>(a + 2);
    // CHECK-NEXT: b = *dpct::get_host_ptr<int>(a) + 1;
    b = a[2];
    b = *a;
    b = *a++;
    b = *++a;
    b = *a--;
    b = *--a;
    b = *(a + 2);
    b = *a + 1;

    // CHECK: dpct::get_host_ptr<int>(a)[2] = b;
    // CHECK-NEXT: *dpct::get_host_ptr<int>(a) = b;
    // CHECK-NEXT: *dpct::get_host_ptr<int>(a++) = b;
    // CHECK-NEXT: *dpct::get_host_ptr<int>(++a) = b;
    // CHECK-NEXT: *dpct::get_host_ptr<int>(a--) = b;
    // CHECK-NEXT: *dpct::get_host_ptr<int>(a + 2) = b;
    a[2] = b;
    *a = b;
    *a++ = b;
    *++a = b;
    *a-- = b;
    *(a + 2) = b;
  };
  ~A1(){
    cudaFree(a);
  }
};

class A2{
  int* a;
  int* b;
  int* c;
public:
  A2(){
    // CHECK: /*
    // CHECK-NEXT: DPCT1070:{{[0-9]+}}: 'a' is allocated by dpct::dpct_malloc. Use 'dpct::get_host_ptr<int>(a)' to access the pointer from the host code.
    // CHECK-NEXT: */
    // CHECK-NEXT: a = (int *)dpct::dpct_malloc(10 * sizeof(int));
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1070:{{[0-9]+}}: 'b' is allocated by dpct::dpct_malloc. Use 'dpct::get_host_ptr<int>(b)' to access the pointer from the host code.
    // CHECK-NEXT: */
    // CHECK-NEXT: b = (int *)dpct::dpct_malloc(10 * sizeof(int));
    cudaMallocManaged(&a, 10 * sizeof(int));
    cudaMallocManaged(&b, 10 * sizeof(int));

  };
  void run2(){

    cudaMalloc(&a, 10 * sizeof(int));

    c = b;
    // CHECK: a[0] = 1;
    a[0] = 1;
    // CHECK: dpct::get_host_ptr<int>(b)[0] = 1;
    b[0] = 1;

  };
  ~A2(){
    cudaFree(a);
    cudaFree(b);
  }
};

class A3{
  A aa;
public:
  A3(){
    A aa;
    // CHECK: /*
    // CHECK-NEXT: DPCT1070:{{[0-9]+}}: 'aa.b.c.a' is allocated by dpct::dpct_malloc. Use 'dpct::get_host_ptr<int>(aa.b.c.a)' to access the pointer from the host code.
    // CHECK-NEXT: */
    // CHECK-NEXT: aa.b.c.a = (int *)dpct::dpct_malloc(10 * sizeof(int));
    cudaMallocManaged(&aa.b.c.a, 10 * sizeof(int));
  }
  void run3(){
    // CHECK: aa.b.c.a[0] = 1;
    aa.b.c.a[0] = 1;
  };
  ~A3(){
    cudaFree(aa.b.c.a);
  }
};

void test1(){
  int *a, b;
  // CHECK: a = (int *)dpct::dpct_malloc(10 * sizeof(int));
  cudaMallocManaged(&a, 10 * sizeof(int));

  // CHECK: b = dpct::get_host_ptr<int>(a)[2];
  // CHECK-NEXT: b = *dpct::get_host_ptr<int>(a);
  // CHECK-NEXT: b = *dpct::get_host_ptr<int>(a++);
  // CHECK-NEXT: b = *dpct::get_host_ptr<int>(++a);
  // CHECK-NEXT: b = *dpct::get_host_ptr<int>(a--);
  // CHECK-NEXT: b = *dpct::get_host_ptr<int>(--a);
  // CHECK-NEXT: b = *dpct::get_host_ptr<int>(a + 2);
  // CHECK-NEXT: b = *dpct::get_host_ptr<int>(a) + 1;
  b = a[2];
  b = *a;
  b = *a++;
  b = *++a;
  b = *a--;
  b = *--a;
  b = *(a + 2);
  b = *a + 1;

  // CHECK: dpct::get_host_ptr<int>(a)[2] = b;
  // CHECK-NEXT: *dpct::get_host_ptr<int>(a) = b;
  // CHECK-NEXT: *dpct::get_host_ptr<int>(a++) = b;
  // CHECK-NEXT: *dpct::get_host_ptr<int>(++a) = b;
  // CHECK-NEXT: *dpct::get_host_ptr<int>(a--) = b;
  // CHECK-NEXT: *dpct::get_host_ptr<int>(a + 2) = b;
  a[2] = b;
  *a = b;
  *a++ = b;
  *++a = b;
  *a-- = b;
  *(a + 2) = b;

  cudaFree(a);
}

void test2(){
  float* a, *b, *c;
  // CHECK: /*
  // CHECK-NEXT: DPCT1070:{{[0-9]+}}: 'a' is allocated by dpct::dpct_malloc. Use 'dpct::get_host_ptr<float>(a)' to access the pointer from the host code.
  // CHECK-NEXT: */
  // CHECK-NEXT: a = (float *)dpct::dpct_malloc(10 * sizeof(float));
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1070:{{[0-9]+}}: 'b' is allocated by dpct::dpct_malloc. Use 'dpct::get_host_ptr<float>(b)' to access the pointer from the host code.
  // CHECK-NEXT: */
  // CHECK-NEXT: b = (float *)dpct::dpct_malloc(10 * sizeof(float));
  cudaMallocManaged(&a, 10 * sizeof(float));
  cudaMallocManaged(&b, 10 * sizeof(float));

  cudaMalloc(&a, 10 * sizeof(float));

  c = b;
  // CHECK: a[0] = 1;
  a[0] = 1;
  // CHECK: dpct::get_host_ptr<float>(b)[0] = 1;
  b[0] = 1;

  cudaFree(a);
  cudaFree(b);
}

void test3(){
  A aa;
  // CHECK: /*
  // CHECK-NEXT: DPCT1070:{{[0-9]+}}: 'aa.b.c.a' is allocated by dpct::dpct_malloc. Use 'dpct::get_host_ptr<int>(aa.b.c.a)' to access the pointer from the host code.
  // CHECK-NEXT: */
  // CHECK-NEXT: aa.b.c.a = (int *)dpct::dpct_malloc(10 * sizeof(int));
  cudaMallocManaged(&aa.b.c.a, 10 * sizeof(int));

  // CHECK: aa.b.c.a[0] = 1;
  aa.b.c.a[0] = 1;

  cudaFree(aa.b.c.a);
}

int main(){
  int *a, *b, *c;

  // CHECK: a = (int *)dpct::dpct_malloc(VECTOR_SIZE * sizeof(float));
  // CHECK-NEXT: b = (int *)dpct::dpct_malloc(VECTOR_SIZE * sizeof(float));
  // CHECK-NEXT: c = (int *)dpct::dpct_malloc(VECTOR_SIZE * sizeof(float));
  cudaMallocManaged(&a, VECTOR_SIZE * sizeof(float));
  cudaMallocManaged(&b, VECTOR_SIZE * sizeof(float));
  cudaMallocManaged(&c, VECTOR_SIZE * sizeof(float));

  for(int i = 0; i < VECTOR_SIZE; i++){
    // CHECK: dpct::get_host_ptr<int>(a)[i] = i;
    // CHECK-NEXT: dpct::get_host_ptr<int>(b)[i] = i;
    a[i] = i;
    b[i] = i;
  }

  ker<<<1, VECTOR_SIZE>>>(a, b, c);

  for(int i = 0; i < VECTOR_SIZE; i++){
    // CHECK: printf("%d", dpct::get_host_ptr<int>(c)[i]);
    printf("%d", c[i]);
  }
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);

  temp1<float>();
  temp2<int>();
  temp3<double>();
  test1();
  test2();
  test3();
  return 0;
}
