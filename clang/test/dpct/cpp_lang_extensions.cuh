
// CHECK: #if (DPCT_COMPATIBILITY_TEMP >= 320)
// CHECK-NEXT: /*
// CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to __ldg was removed because there is no correspoinding API in SYCL.
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
