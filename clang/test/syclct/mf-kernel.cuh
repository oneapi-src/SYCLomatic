#include <cuda_runtime.h>

  // CHECK: void Reset_kernel_parameters(syclct::syclct_accessor<volatile int, syclct::device, 0> g_mutex);
__global__ void Reset_kernel_parameters(void);
