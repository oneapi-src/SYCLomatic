// RUN: dpct --format-range=none -out-root %T/device_usmnone %s --cuda-include-path="%cuda-path/include" --usm-level=none -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/device_usmnone/device_usmnone.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/device_usmnone/device_usmnone.dp.cpp -o %T/device_usmnone/device_usmnone.dp.o %}
#include <cuda.h>

int main() {
int concurrentManagedAccess = 0;
int p_gpuDevice = 0;
// CHECK: int error = DPCT_CHECK_ERROR(concurrentManagedAccess = false);
int error = cudaDeviceGetAttribute(&concurrentManagedAccess,  cudaDevAttrConcurrentManagedAccess,  p_gpuDevice);
}