// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none --optimize-migration -out-root %T/optimize %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --input-file %T/optimize/optimize.dp.cpp

// clang-format off
#include <cuda_runtime.h>

// CHECK: void test1() {
// CHECK-NEXT: int a = 0, b = 1;
// CHECK-NEXT: #if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
// CHECK-NEXT:   asm("mov.s32 %0, %1;" : "=r"(a) : "r"(b));
// CHECK-NEXT: #else
// CHECK-NEXT:   a = b;
// CHECK-NEXT: #endif
// CHECK: }
__global__ void test1() {
  int a = 0, b = 1;
  asm("mov.s32 %0, %1;" : "=r"(a) : "r"(b));
}

// CHECK: void test2() {
// CHECK-NEXT: #if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
// CHECK-NEXT: #define TEST2(X, Y)
// CHECK-NEXT:   asm("{\n\t"
// CHECK-NEXT:       " .reg .pred p;\n\t"
// CHECK-NEXT:       " setp.eq.s32 p, %1, 34;\n\t"
// CHECK-NEXT:       " @p mov.s32 %0, 1;\n\t"
// CHECK-NEXT:       "}"
// CHECK-NEXT:       : "+r"(Y) : "r" (X))
// CHECK-NEXT: #else
// CHECK-NEXT: #define TEST2(X, Y)
// CHECK-NEXT:   {
// CHECK-NEXT:     bool p;
// CHECK-NEXT:     p = X == 34;
// CHECK-NEXT:     if (p) {
// CHECK-NEXT:       Y = 1;
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: #endif
// CHECK: int x = 34;
// CHECK-NEXT: int y = 0;
// CHECK-NEXT: COND2(x, y);
// CHECK-NEXT: }
__global__ void test2() {
#define TEST2(X, Y)                 \
  asm("{\n\t"                       \
      " .reg .pred p;\n\t"          \
      " setp.eq.s32 p, %1, 34;\n\t" \
      " @p mov.s32 %0, 1;\n\t"      \
      "}"                           \
      : "+r"(Y) : "r" (X))
  int x = 34;
  int y = 0;
  TEST2(x, y);
}

// CHECK: void test3() {
// CHECK-NEXT: #if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
// CHECK-NEXT: #define TEST3(V)               
// CHECK-NEXT:   asm("{\n\t"                 
// CHECK-NEXT:       " .reg .u32 p<10>;\n\t" 
// CHECK-NEXT:       " mov.u32 p0, 2013;\n\t"
// CHECK-NEXT:       " mov.u32 p1, p0;\n\t"  
// CHECK-NEXT:       " mov.u32 p2, p1;\n\t"  
// CHECK-NEXT:       " mov.u32 p3, p2;\n\t"  
// CHECK-NEXT:       " mov.u32 p4, p3;\n\t"  
// CHECK-NEXT:       " mov.u32 p5, p4;\n\t"  
// CHECK-NEXT:       " mov.u32 p6, p5;\n\t"  
// CHECK-NEXT:       " mov.u32 p7, p6;\n\t"  
// CHECK-NEXT:       " mov.u32 p8, p7;\n\t"  
// CHECK-NEXT:       " mov.u32 p9, p8;\n\t"  
// CHECK-NEXT:       " mov.u32 %0, p9;\n\t"  
// CHECK-NEXT:       "}"                     
// CHECK-NEXT:       : "=r"(V))
// CHECK-NEXT: #else
// CHECK-NEXT: #define TEST3(V)
// CHECK-NEXT:   {
// CHECK-NEXT:     uint32_t p[10];
// CHECK-NEXT:     p[0] = 2013;
// CHECK-NEXT:     p[1] = p[0];
// CHECK-NEXT:     p[2] = p[1];
// CHECK-NEXT:     p[3] = p[2];
// CHECK-NEXT:     p[4] = p[3];
// CHECK-NEXT:     p[5] = p[4];
// CHECK-NEXT:     p[6] = p[5];
// CHECK-NEXT:     p[7] = p[6];
// CHECK-NEXT:     p[8] = p[7];
// CHECK-NEXT:     p[9] = p[8];
// CHECK-NEXT:     V = p[9];
// CHECK-NEXT:   }
// CHECK-NEXT: #endif
// CHECK-NEXT: int x = 34;
// CHECK-NEXT: TEST3(x);
// CHECK-NEXT: }
__global__ void test3() {
#define TEST3(V)                     \
  asm("{\n\t"                       \
      " .reg .u32 p<10>;\n\t"       \
      " mov.u32 p0, 2013;\n\t"      \
      " mov.u32 p1, p0;\n\t"        \
      " mov.u32 p2, p1;\n\t"        \
      " mov.u32 p3, p2;\n\t"        \
      " mov.u32 p4, p3;\n\t"        \
      " mov.u32 p5, p4;\n\t"        \
      " mov.u32 p6, p5;\n\t"        \
      " mov.u32 p7, p6;\n\t"        \
      " mov.u32 p8, p7;\n\t"        \
      " mov.u32 p9, p8;\n\t"        \
      " mov.u32 %0, p9;\n\t"        \
      "}"                           \
      : "=r"(V))
  int x = 34;
  TEST3(x);
}

// CHECK: void test4() {
// CHECK-NEXT: #if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
// CHECK-NEXT: #define TEST4(X, Y)
// CHECK-NEXT:   asm("{\n\t"
// CHECK-NEXT:       " .reg .pred p;\n\t"
// CHECK-NEXT:       " setp.eq.s32 p, %1, 34;\n\t"
// CHECK-NEXT:       " @p mov.s32 %0, 1;\n\t"
// CHECK-NEXT:       "}"
// CHECK-NEXT:       : "+r"(Y) : "r" (X))
// CHECK-NEXT: #else
// CHECK-NEXT: #define TEST4(X, Y)
// CHECK-NEXT:   {
// CHECK-NEXT:     bool p;
// CHECK-NEXT:     p = X == 34;
// CHECK-NEXT:     if (p) {
// CHECK-NEXT:       Y = 1;
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: #endif
// CHECK-NEXT: int x = 34;
// CHECK-NEXT: int y = 0;
// CHECK-NEXT: TEST4(x, y);
// CHECK-NEXT: }
__global__ void test4() {
#define TEST4(X, Y)                   \
  {                                   \
    X = 0;                            \
    asm("{\n\t"                       \
        " .reg .pred p;\n\t"          \
        " setp.eq.s32 p, %1, 34;\n\t" \
        " @p mov.s32 %0, 1;\n\t"      \
        "}"                           \
        : "+r"(Y) : "r" (X));         \
    X = X * X;                        \
  }
  int x = 34;
  int y = 0;
  TEST4(x, y);
}

#define MACRO(ID, CMP, S) \
  {                       \
    S;                    \
    if (!(CMP)) {         \
      return ID;          \
    }                     \
  }

// CHECK: int test5() {
// CHECK-NEXT:   int s32 = 0, s32x = 1;
// CHECK-NEXT: #if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
// CHECK-NEXT:   MACRO(7, s32 == 0, asm("add.s32.sat %0, %1, %2;" : "=r"(s32) : "r"(s32x), "r"(0)));
// CHECK-NEXT: #else
// CHECK-NEXT:   MACRO(7, s32 == 0, s32 = sycl::add_sat((int32_t)s32x, (int32_t)0));
// CHECK-NEXT: #endif
// CHECK-NEXT:   return 0;
// CHECK-NEXT: }
__device__ int test5() {
  int s32 = 0, s32x = 1;
  MACRO(7, s32 == 0, asm("add.s32.sat %0, %1, %2;" : "=r"(s32) : "r"(s32x), "r"(0)));
  return 0;
}
