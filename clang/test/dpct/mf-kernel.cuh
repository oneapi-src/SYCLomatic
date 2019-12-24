#include <cuda_runtime.h>

// CHECK: SYCL_EXTERNAL void Reset_kernel_parameters(volatile int *g_mutex);
__global__ void Reset_kernel_parameters(void);

// CHECK: SYCL_EXTERNAL void test_foo();
__device__ void test_foo(void);
