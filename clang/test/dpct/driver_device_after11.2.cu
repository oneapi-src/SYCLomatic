// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1
// RUN: dpct --format-range=none -out-root %T/driver_device_after11.2 %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/driver_device_after11.2/driver_device_after11.2.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/driver_device_after11.2/driver_device_after11.2.dp.cpp -o %T/driver_device_after11.2/driver_device_after11.2.dp.o %}

#include <cuda.h>

int main() {
  int i;
  CUdevice d;
  // CHECK: i = dpct::get_device(d).has(sycl::aspect::ext_oneapi_virtual_mem);
  cuDeviceGetAttribute(&i, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, d);
  return 0;
}
