
// CHECK: #if (DPCT_COMPATIBILITY_TEMP >= 320)
// CHECK-NEXT: /*
// CHECK-NEXT: DPCT1098:{{[0-9]+}}: The '*' expression is used instead of the __ldg call. These two expressions do not provide the exact same functionality. Check the generated code for potential precision and/or performance issues.
// CHECK-NEXT: */
// CHECK-NEXT: #define LDG(x) (x)
// CHECK-NEXT: #else
// CHECK-NEXT: #define LDG(x) (x)
// CHECK-NEXT: #endif
#if (__CUDA_ARCH__ >= 320)
#define LDG(x) __ldg(&(x))
#else
#define LDG(x) (x)
#endif
