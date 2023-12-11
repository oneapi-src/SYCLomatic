// RUN: dpct --format-range=none -out-root %T/device_usm %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/device_usm/device_usm.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/device_usm/device_usm.dp.cpp -o %T/device_usm/device_usm.dp.o %}
#include <cuda.h>

int main() {
int concurrentManagedAccess = 0;
int p_gpuDevice = 0;
// CHECK: int error = DPCT_CHECK_ERROR(concurrentManagedAccess = dpct::dev_mgr::instance().get_device(p_gpuDevice).get_info<sycl::info::device::usm_shared_allocations>());
int error = cudaDeviceGetAttribute(&concurrentManagedAccess,  cudaDevAttrConcurrentManagedAccess,  p_gpuDevice);
}