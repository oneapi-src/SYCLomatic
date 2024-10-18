// RUN: dpct --format-range=none --optimize-migration -out-root %T/opt_loop_unroll %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --match-full-lines --input-file %T/opt_loop_unroll/opt_loop_unroll.dp.cpp %s
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST  %T/opt_loop_unroll/opt_loop_unroll.dp.cpp -o %T/opt_loop_unroll/opt_loop_unroll.dp.o %}
#ifndef NO_BUILD_TEST
#include <cuda_runtime.h>

__global__ void kernel(float *a, float *b, float *c){
// CHECK:  #pragma unroll
// CHECK:  for(int i = 0; i < 10; i++) {
// CHECK:    c[i] = a[i] + b[i];
// CHECK:  }
    for(int i = 0; i < 10; i++) {
      c[i] = a[i] + b[i];
    }

// CHECK:  #pragma unroll
// CHECK:  for(int i = 0; i < 10; i++) {
// CHECK:    c[i] = a[i] + b[i];
// CHECK:  }
    #pragma unroll
    for(int i = 0; i < 10; i++) {
      c[i] = a[i] + b[i];
    }

// CHECK:  int p = 0;
// CHECK-NOT: #pragma unroll
// CHECK-NEXT:  for(int i = 0; i < 10; i++) {
// CHECK-NEXT:    int d = 1;
// CHECK-NEXT:    c[i] = a[i] + b[i] + d;
// CHECK-NEXT:  }
    int p = 0;
    for(int i = 0; i < 10; i++) {
      int d = 1;
      c[i] = a[i] + b[i] + d;
    }

// CHECK:  for(int i = 0; i < 10; i++) {
// CHECK:      int d = 1;
// CHECK:      c[i] = a[i] + b[i] + d;
// CHECK:      #pragma unroll
// CHECK:      for(int j = 0; j < 9; j++){
// CHECK:        c[j] = a[j] + b[j];
// CHECK:      }
// CHECK:   }
    for(int i = 0; i < 10; i++) {
      int d = 1;
      c[i] = a[i] + b[i] + d;
      for(int j = 0; j < 9; j++){
        c[j] = a[j] + b[j];  
      }
    }

    bool n;
// CHECK: if(n)
// CHECK-NEXT: #pragma unroll 
// CHECK-NEXT: for(int i = 0; i < 10; i++) { c[i] = 0; }
    if(n) for(int i = 0; i < 10; i++) { c[i] = 0; }
}

__device__ void bar(float *a, float *b, float *c) {
  // CHECK: int i = 0;
  // CHECK-NEXT:  #pragma unroll
  // CHECK-NEXT:  for(i = 0; i < 10; i++) {
  // CHECK-NEXT:    c[i] = a[i] + b[i];
  // CHECK-NEXT:  }
    int i = 0;
    for(i = 0; i < 10; i++) {
      c[i] = a[i] + b[i];
    }
}

void foo1(float *a, float *b, float *c) {
// CHECK: int i = 0;
// CHECK-NOT:  #pragma unroll
// CHECK-NEXT:  for(i = 0; i < 10; i++) {
// CHECK-NEXT:    c[i] = a[i] + b[i];
// CHECK-NEXT:  }
  int i = 0;
  for(i = 0; i < 10; i++) {
    c[i] = a[i] + b[i];
  }
}

class A{
 int j = 0;
public:
 __device__ void a(float *a, float *b, float *c){
  // CHECK: int i = 0;
  // CHECK-NEXT:  #pragma unroll
  // CHECK-NEXT:  for(i = 0; i < 10; i++) {
  // CHECK-NEXT:    c[i] = a[i] + b[i];
  // CHECK-NEXT:  }
  int i = 0;
  for(i = 0; i < 10; i++) {
    c[i] = a[i] + b[i];
  }
 }
}

// CHECK: #define copy(dst, src, count)          \
// CHECK-NEXT:     for (int i = 0; i != count; ++i) { \
// CHECK-NEXT:         (dst)[i] = (src)[i];           \
// CHECK-NEXT:     }
#define copy(dst, src, count)          \
    for (int i = 0; i != count; ++i) { \
        (dst)[i] = (src)[i];           \
    }
__device__ void bar(float *a, float *b) {
  // CHECK: copy(a, b, 10)
  copy(a, b, 10)
}

int main(){
    float *a, *b, *c;
    kernel<<<10, 10>>>(a, b, c);
    return 0;
}
#endif
