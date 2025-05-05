// RUN: echo "empty command"


// CHECK: SYCL_EXTERNAL void kernel_extern(int *a);
__global__ void kernel_extern();