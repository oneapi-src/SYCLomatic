#include <cuda_runtime.h>

// CHECK: SYCL_EXTERNAL void Reset_kernel_parameters(dpct::accessor<volatile int, dpct::device, 0> g_mutex);
__global__ void Reset_kernel_parameters(void);

// CHECK: SYCL_EXTERNAL void test_foo();
__device__ void test_foo(void);
