// RUN: dpct --format-range=none -out-root %T/device_usmnone %s --cuda-include-path="%cuda-path/include" --usm-level=none -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/device_usmnone/device_usmnone.dp.cpp
#include <cuda.h>

int main() {
int concurrentManagedAccess = 0;
int p_gpuDevice = 0;
// CHECK: int error = CHECK_SYCL_ERROR(concurrentManagedAccess = false);
int error = cudaDeviceGetAttribute(&concurrentManagedAccess,  cudaDevAttrConcurrentManagedAccess,  p_gpuDevice);
}