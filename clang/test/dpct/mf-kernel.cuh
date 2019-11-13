#include <cuda_runtime.h>

  // CHECK: void Reset_kernel_parameters(dpct::accessor<volatile int, dpct::device, 0> g_mutex);
__global__ void Reset_kernel_parameters(void);
