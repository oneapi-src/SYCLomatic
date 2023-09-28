// RUN: dpct --format-range=none -out-root %T/device003 %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/device003/device003.dp.cpp

#include<cuda_runtime.h>
#include<cstdio>

template <typename T>
void check(T result, char const *const func) {}

#define checkErrors(val) check((val), #val)

#define CUDA_SAFE_CALL( call) do {                                         \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
      fprintf(stderr, "Cuda error in call at file '%s' in line %i : %s.\n",  \
       __FILE__, __LINE__, cudaGetErrorString( err) );                       \
    } } while (0)

int main(int argc, char **argv)
{
int deviceCount = 0;

// CHECK: checkErrors(DPCT_CHECK_ERROR(deviceCount = dpct::dev_mgr::instance().device_count()));
checkErrors(cudaGetDeviceCount(&deviceCount));

int dev_id;
// CHECK: checkErrors(dev_id = dpct::dev_mgr::instance().current_device_id());
checkErrors(cudaGetDevice(&dev_id));

cudaDeviceProp deviceProp;
// CHECK:checkErrors(DPCT_CHECK_ERROR(dpct::get_device_info(deviceProp, dpct::dev_mgr::instance().get_device(0))));
checkErrors(cudaGetDeviceProperties(&deviceProp, 0));

// CHECK:CUDA_SAFE_CALL(DPCT_CHECK_ERROR(dpct::get_device_info(deviceProp, dpct::dev_mgr::instance().get_device(dev_id))));
CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev_id));

int atomicSupported;
// CHECK: checkErrors(DPCT_CHECK_ERROR(atomicSupported = dpct::dev_mgr::instance().get_device(dev_id).is_native_atomic_supported()));
checkErrors(cudaDeviceGetAttribute(&atomicSupported, cudaDevAttrHostNativeAtomicSupported, dev_id));

int maxThreadsPerBlock;
// CHECK: checkErrors(DPCT_CHECK_ERROR(maxThreadsPerBlock = dpct::dev_mgr::instance().get_device(0).get_max_work_group_size()));
checkErrors(cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0));

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
// CHECK-NEXT:DPCT1093:{{[0-9]+}}: The "device2" device may be not the one intended for use. Adjust the selected device if needed.
// CHECK-NEXT:*/
// CHECK-NEXT:checkErrors(DPCT_CHECK_ERROR(dpct::select_device(device2)));
checkErrors(cudaSetDevice(device2));

return 0;
}

void get_version(void) {
    int driverVersion, runtimeVersion;

    // CHECK: /*
    // CHECK-NEXT:  DPCT1043:{{[0-9]+}}: The version-related API is different in SYCL. An initial code was generated, but you need to adjust it.
    // CHECK-NEXT:  */
    // CHECK-NEXT:  driverVersion = dpct::get_major_version(dpct::get_current_device());
    cudaDriverGetVersion(&driverVersion);

    // CHECK:  /*
    // CHECK-NEXT:  DPCT1043:{{[0-9]+}}: The version-related API is different in SYCL. An initial code was generated, but you need to adjust it.
    // CHECK-NEXT:  */
    // CHECK-NEXT:  runtimeVersion = dpct::get_major_version(dpct::get_current_device());
    cudaRuntimeGetVersion(&runtimeVersion);

    // CHECK:    /*
    // CHECK-NEXT:    DPCT1043:{{[0-9]+}}: The version-related API is different in SYCL. An initial code was generated, but you need to adjust it.
    // CHECK-NEXT:    */
    // CHECK-NEXT:    dpct::err0 error_code_1 = DPCT_CHECK_ERROR(driverVersion = dpct::get_major_version(dpct::get_current_device()));
    cudaError_t error_code_1 = cudaDriverGetVersion(&driverVersion);

    // CHECK:    /*
    // CHECK-NEXT:    DPCT1043:{{[0-9]+}}: The version-related API is different in SYCL. An initial code was generated, but you need to adjust it.
    // CHECK-NEXT:    */
    // CHECK-NEXT:    dpct::err0 error_code_2 = DPCT_CHECK_ERROR(runtimeVersion = dpct::get_major_version(dpct::get_current_device()));
    cudaError_t error_code_2 = cudaRuntimeGetVersion(&runtimeVersion);
}

