// RUN: dpct --format-range=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/device002.dp.cpp

#include <stdio.h>

void checkError(cudaError_t err) {

}

int main(int argc, char **argv)
{
int devID = atoi(argv[1]);
cudaDeviceProp cdp;
// CHECK:/*
// CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
// CHECK-NEXT:*/
// CHECK-NEXT: int error_code = (dpct::dev_mgr::instance().get_device(devID).get_device_info(cdp), 0);
cudaError_t error_code = cudaGetDeviceProperties(&cdp, devID);

if (error_code == cudaSuccess) {
// CHECK: /*
// CHECK-NEXT:  DPCT1005:{{[0-9]+}}: The device version is different. You need to rewrite this code.
// CHECK-NEXT: */
// CHECK-NEXT: /*
// CHECK-NEXT:  DPCT1006:{{[0-9]+}}: DPC++ does not provide a standard API to differentiate between integrated/ discrete GPU devices.
// CHECK-NEXT: */
// CHECK-NEXT:if (cdp.get_major_version() < 3 && cdp.get_integrated() != 1) {
    if (cdp.major < 3 && cdp.integrated != 1) {
            printf("do_complex_compute requires compute capability 3.0 or later and not integrated\n");
    }
}

int deviceCount = 0;
// CHECK: deviceCount = dpct::dev_mgr::instance().device_count();
cudaGetDeviceCount(&deviceCount);

int dev_id;
// CHECK: dev_id = dpct::dev_mgr::instance().current_device_id();
cudaGetDevice(&dev_id);

cudaDeviceProp deviceProp;
// CHECK: dpct::dev_mgr::instance().get_device(0).get_device_info(deviceProp);
cudaGetDeviceProperties(&deviceProp, 0);

int atomicSupported;
// CHECK: atomicSupported = dpct::dev_mgr::instance().get_device(dev_id).is_native_atomic_supported();
cudaDeviceGetAttribute(&atomicSupported, cudaDevAttrHostNativeAtomicSupported, dev_id);

int val;
// CHECK: val = dpct::dev_mgr::instance().get_device(dev_id).get_major_version();
cudaDeviceGetAttribute(&val, cudaDevAttrComputeCapabilityMajor, dev_id);

int computeMode = -1, minor = 0;
// CHECK: /*
// CHECK-NEXT: DPCT1035:{{[0-9]+}}: All DPC++ devices can be used by host to submit tasks. You may need to adjust this code.
// CHECK-NEXT: */
// CHECK-NEXT: checkError((computeMode = 1, 0));
checkError(cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, dev_id));
// CHECK: checkError((minor = dpct::dev_mgr::instance().get_device(dev_id).get_minor_version(), 0));
checkError(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev_id));

int multiProcessorCount = 0, clockRate = 0;
// CHECK: checkError((multiProcessorCount = dpct::dev_mgr::instance().get_device(dev_id).get_max_compute_units(), 0));
checkError(cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, dev_id));
// CHECK: checkError((clockRate = dpct::dev_mgr::instance().get_device(dev_id).get_max_clock_frequency(), 0));
checkError(cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, dev_id));

int integrated = -1;
// CHECK: checkError((integrated = dpct::dev_mgr::instance().get_device(dev_id).get_integrated(), 0));
checkError(cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, dev_id));

int device1 = 0;
int device2 = 1;
int perfRank = 0;
int accessSupported = 0;

// CHECK:/*
// CHECK-NEXT:DPCT1004:{{[0-9]+}}: Could not generate replacement.
// CHECK-NEXT:*/
// CHECK-NEXT: accessSupported = 0;
cudaDeviceGetP2PAttribute(&accessSupported, cudaDevP2PAttrAccessSupported, device1, device2);

// CHECK:/*
// CHECK-NEXT:DPCT1004:{{[0-9]+}}: Could not generate replacement.
// CHECK-NEXT:*/
// CHECK-NEXT: perfRank = 0;
cudaDeviceGetP2PAttribute(&perfRank, cudaDevP2PAttrPerformanceRank, device1, device2);

// CHECK:/*
// CHECK-NEXT:DPCT1004:{{[0-9]+}}: Could not generate replacement.
// CHECK-NEXT:*/
// CHECK-NEXT: atomicSupported = 0;
cudaDeviceGetP2PAttribute(&atomicSupported, cudaDevP2PAttrNativeAtomicSupported, device1, device2);


char pciBusId[80];
// CHECK:/*
// CHECK-NEXT:DPCT1004:{{[0-9]+}}: Could not generate replacement.
// CHECK-NEXT:*/
cudaDeviceGetPCIBusId(pciBusId, 80, 0);


// CHECK: dpct::get_current_device().reset();
cudaDeviceReset();

// CHECK:/*
// CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
// CHECK-NEXT:*/
// CHECK-NEXT:error_code = (dpct::get_current_device().reset(), 0);
error_code = cudaDeviceReset();

// CHECK: dpct::get_current_device().reset();
cudaThreadExit();

// CHECK:/*
// CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
// CHECK-NEXT:*/
// CHECK-NEXT:error_code = (dpct::get_current_device().reset(), 0);
error_code = cudaThreadExit();

// CHECK:/*
// CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
// CHECK-NEXT:*/
// CHECK-NEXT:error_code = (dpct::dev_mgr::instance().select_device(device2), 0);
error_code = cudaSetDevice(device2);
// CHECK: dpct::dev_mgr::instance().select_device(device2);
cudaSetDevice(device2);

// CHECK:dpct::get_current_device().queues_wait_and_throw();
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
// CHECK-NEXT:*/
// CHECK-NEXT:int err = (dpct::get_current_device().queues_wait_and_throw(), 0);
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
// CHECK-NEXT:*/
// CHECK-NEXT:checkError((dpct::get_current_device().queues_wait_and_throw(), 0));
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
// CHECK-NEXT:*/
// CHECK-NEXT:return (dpct::get_current_device().queues_wait_and_throw(), 0);
cudaDeviceSynchronize();
cudaError_t err = cudaDeviceSynchronize();
checkError(cudaDeviceSynchronize());
return cudaDeviceSynchronize();
// CHECK:/*
// CHECK-NEXT:DPCT1010:{{[0-9]+}}: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
// CHECK-NEXT:*/
// CHECK-NEXT: int e = 0;
int e = cudaGetLastError();
// CHECK:/*
// CHECK-NEXT:DPCT1010:{{[0-9]+}}: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
// CHECK-NEXT:*/
// CHECK-NEXT: int e1 = 0;
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1010:{{[0-9]+}}: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
// CHECK-NEXT:*/
// CHECK-NEXT: 0;
int e1 = cudaPeekAtLastError();
cudaPeekAtLastError();
// CHECK:dpct::get_current_device().queues_wait_and_throw();
cudaThreadSynchronize();
return 0;
}
