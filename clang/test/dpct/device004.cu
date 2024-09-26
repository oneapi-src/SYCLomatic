// UNSUPPORTED: cuda-8.0, cuda-9.1, cuda-9.2
// UNSUPPORTED: v8.0, v9.1, v9.2
// RUN: dpct --no-dpcpp-extensions=device_info -out-root %T/device004 %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/device004/device004.dp.cpp

#include <cuda.h>
#include <iostream>

int main() {
  cudaDeviceProp properties;
  cudaGetDeviceProperties(&properties, 0);

  // CHECK: /*
  // CHECK-NEXT: DPCT1090:{{[0-9]+}}: SYCL does not support the device property that would be functionally compatible with pciDeviceID. It was not migrated. You need to rewrite the code.
  // CHECK-NEXT: */
  // CHECK-NEXT: const int id = properties.pciDeviceID;
  const int id = properties.pciDeviceID;
  // CHECK: /*
  // CHECK-NEXT: DPCT1090:{{[0-9]+}}: SYCL does not support the device property that would be functionally compatible with uuid. It was not migrated. You need to rewrite the code.
  // CHECK-NEXT: */
  // CHECK-NEXT: const std::array<unsigned char, 16> uuid = properties.uuid;
  const cudaUUID_t uuid = properties.uuid;
  // CHECK: /*
  // CHECK-NEXT: DPCT1090:{{[0-9]+}}: SYCL does not support the device property that would be functionally compatible with pciDeviceID. It was not migrated. You need to rewrite the code.
  // CHECK-NEXT: */
  // CHECK-NEXT: properties.pciDeviceID = id;
  properties.pciDeviceID = id;
  // CHECK: /*
  // CHECK-NEXT: DPCT1090:{{[0-9]+}}: SYCL does not support the device property that would be functionally compatible with uuid. It was not migrated. You need to rewrite the code.
  // CHECK-NEXT: */
  // CHECK-NEXT: properties.uuid = uuid;
  properties.uuid = uuid;
}

#define CHECK_INTERNAL(err)                                                    \
  { auto err_ = (err); }

#define CHECK(err) CHECK_INTERNAL(err)

void foo() {
  int dev = 1;
  cudaDeviceProp p;
  // CHECK: CHECK(DPCT_CHECK_ERROR(dpct::get_device(dev).get_device_info(p)));
  CHECK(cudaGetDeviceProperties(&p, dev));
}
