
// CHECK: #include <sycl/ext/oneapi/experimental/cuda/builtins.hpp>
// CHECK: #if (DPCT_COMPATIBILITY_TEMP >= 320)
// CHECK: #define LDG(x) sycl::ext::oneapi::experimental::cuda::ldg(&(x));
// CHECK-NEXT: #else
// CHECK-NEXT: #define LDG(x) (x)
// CHECK-NEXT: #endif
#if (__CUDA_ARCH__ >= 320)
#define LDG(x) __ldg(&(x))
#else
#define LDG(x) (x)
#endif
