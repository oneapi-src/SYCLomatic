// RUN: dpct --format-range=none -out-root %T/device003 %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/device003/device003.dp.cpp

template <typename T>
void check(T result, char const *const func) {}

#define checkErrors(val) check((val), #val)

int main(int argc, char **argv)
{
int deviceCount = 0;

// CHECK:/*
// CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
// CHECK-NEXT:*/
// CHECK-NEXT: checkErrors((deviceCount = dpct::dev_mgr::instance().device_count(), 0));
checkErrors(cudaGetDeviceCount(&deviceCount));

int dev_id;
// CHECK: checkErrors(dev_id = dpct::dev_mgr::instance().current_device_id());
checkErrors(cudaGetDevice(&dev_id));

cudaDeviceProp deviceProp;
// CHECK:/*
// CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
// CHECK-NEXT:*/
// CHECK-NEXT:checkErrors((dpct::dev_mgr::instance().get_device(0).get_device_info(deviceProp), 0));
checkErrors(cudaGetDeviceProperties(&deviceProp, 0));

int atomicSupported;
// CHECK: checkErrors((atomicSupported = dpct::dev_mgr::instance().get_device(dev_id).is_native_atomic_supported(), 0));
checkErrors(cudaDeviceGetAttribute(&atomicSupported, cudaDevAttrHostNativeAtomicSupported, dev_id));

int device1 = 0;
int device2 = 1;
int perfRank = 0;
int accessSupported = 0;

// CHECK:/*
// CHECK-NEXT:DPCT1007:{{[0-9]+}}: Migration of cudaDeviceGetP2PAttribute is not supported.
// CHECK-NEXT:*/
// CHECK-NEXT: checkErrors(accessSupported = 0);
checkErrors(cudaDeviceGetP2PAttribute(&accessSupported, cudaDevP2PAttrAccessSupported, device1, device2));

// CHECK:/*
// CHECK-NEXT:DPCT1007:{{[0-9]+}}: Migration of cudaDeviceGetP2PAttribute is not supported.
// CHECK-NEXT:*/
// CHECK-NEXT: checkErrors(perfRank = 0);
checkErrors(cudaDeviceGetP2PAttribute(&perfRank, cudaDevP2PAttrPerformanceRank, device1, device2));

// CHECK:/*
// CHECK-NEXT:DPCT1007:{{[0-9]+}}: Migration of cudaDeviceGetP2PAttribute is not supported.
// CHECK-NEXT:*/
// CHECK-NEXT: checkErrors(atomicSupported = 0);
checkErrors(cudaDeviceGetP2PAttribute(&atomicSupported, cudaDevP2PAttrNativeAtomicSupported, device1, device2));
// CHECK:/*
// CHECK-NEXT:DPCT1093:{{[0-9]+}}: The "device2" may not be the best XPU device. Adjust the selected device if needed.
// CHECK-NEXT:*/
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
// CHECK-NEXT:*/
// CHECK-NEXT:checkErrors((dpct::dev_mgr::instance().select_device(device2), 0));
checkErrors(cudaSetDevice(device2));

return 0;
}

void get_version(void) {
    int driverVersion, runtimeVersion;

    // CHECK: /*
    // CHECK-NEXT:  DPCT1043:{{[0-9]+}}: The version-related API is different in SYCL. An initial code was generated, but you need to adjust it.
    // CHECK-NEXT:  */
    // CHECK-NEXT:  driverVersion = dpct::get_current_device().get_info<sycl::info::device::version>();
    cudaDriverGetVersion(&driverVersion);

    // CHECK:  /*
    // CHECK-NEXT:  DPCT1043:{{[0-9]+}}: The version-related API is different in SYCL. An initial code was generated, but you need to adjust it.
    // CHECK-NEXT:  */
    // CHECK-NEXT:  runtimeVersion = dpct::get_current_device().get_info<sycl::info::device::version>();
    cudaRuntimeGetVersion(&runtimeVersion);

    // CHECK:    /*
    // CHECK-NEXT:    DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT:    */
    // CHECK-NEXT:    /*
    // CHECK-NEXT:    DPCT1043:{{[0-9]+}}: The version-related API is different in SYCL. An initial code was generated, but you need to adjust it.
    // CHECK-NEXT:    */
    // CHECK-NEXT:    int error_code_1 = (driverVersion = dpct::get_current_device().get_info<sycl::info::device::version>(), 0);
    cudaError_t error_code_1 = cudaDriverGetVersion(&driverVersion);

    // CHECK:    /*
    // CHECK-NEXT:    DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT:    */
    // CHECK-NEXT:    /*
    // CHECK-NEXT:    DPCT1043:{{[0-9]+}}: The version-related API is different in SYCL. An initial code was generated, but you need to adjust it.
    // CHECK-NEXT:    */
    // CHECK-NEXT:    int error_code_2 = (runtimeVersion = dpct::get_current_device().get_info<sycl::info::device::version>(), 0);
    cudaError_t error_code_2 = cudaRuntimeGetVersion(&runtimeVersion);
}

