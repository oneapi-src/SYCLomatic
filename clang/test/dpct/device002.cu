// RUN: dpct --format-range=none -out-root %T/device002 %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/device002/device002.dp.cpp

#include <stdio.h>

void checkError(cudaError_t err) {

}

cudaDeviceAttr attr6;

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
// CHECK-NEXT:  DPCT1005:{{[0-9]+}}: The SYCL device version is different from CUDA Compute Compatibility. You may need to rewrite this code.
// CHECK-NEXT: */
// CHECK-NEXT: /*
// CHECK-NEXT:  DPCT1006:{{[0-9]+}}: SYCL does not provide a standard API to differentiate between integrated and discrete GPU devices.
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

struct attr{
    cudaDeviceAttr attr;
} attr1;
// CHECK: /*
// CHECK-NEXT: DPCT1076:{{[0-9]+}}: The device attribute was not recognized. You may need to adjust the code.
// CHECK-NEXT: */
// CHECK_NEXT: cudaDeviceGetAttribute(&result, attr1.attr, dev_id);
cudaDeviceGetAttribute(&val, attr1.attr, dev_id);

// CHECK: int attr2 = 86;
// CHECK-NEXT: atomicSupported = dpct::dev_mgr::instance().get_device(dev_id).is_native_atomic_supported();
cudaDeviceAttr attr2 = cudaDevAttrHostNativeAtomicSupported;
cudaDeviceGetAttribute(&atomicSupported, attr2, dev_id);

// CHECK: int attr3;
// CHECK-NEXT: attr3 = 75;
// CHECK-NEXT: val = dpct::dev_mgr::instance().get_device(dev_id).get_major_version();
cudaDeviceAttr attr3;
attr3 = cudaDevAttrComputeCapabilityMajor;
cudaDeviceGetAttribute(&val, attr3, dev_id);

// CHECK: int attr4;
// CHECK-NEXT: attr4 = 86;
// CHECK-NEXT: attr4 = 75;
// CHECK-NEXT: val = dpct::dev_mgr::instance().get_device(dev_id).get_major_version();
cudaDeviceAttr attr4;
attr4 = cudaDevAttrHostNativeAtomicSupported;
attr4 = cudaDevAttrComputeCapabilityMajor;
cudaDeviceGetAttribute(&val, attr4, dev_id);

// CHECK: int attr5;
// CHECK-NEXT: int somecondition;
// CHECK-NEXT: attr5 = 86;
// CHECK-NEXT: attr5 = 75;
// CHECK-NEXT: if(somecondition)
// CHECK-NEXT:   attr5 = 86;
// CHECK-NEXT: /*
// CHECK-NEXT: DPCT1076:{{[0-9]+}}: The device attribute was not recognized. You may need to adjust the code.
// CHECK-NEXT: */
// CHECK_NEXT: cudaDeviceGetAttribute(&result, attr5, dev_id);
cudaDeviceAttr attr5;
int somecondition;
attr5 = cudaDevAttrHostNativeAtomicSupported;
attr5 = cudaDevAttrComputeCapabilityMajor;
if(somecondition)
  attr5 = cudaDevAttrHostNativeAtomicSupported;
cudaDeviceGetAttribute(&val, attr5, dev_id);

// CHECK: attr5 = 86;
// CHECK-NEXT: attr6 = attr5;
// CHECK-NEXT: checkError((val = dpct::dev_mgr::instance().get_device(dev_id).is_native_atomic_supported(), 0));
attr5 = cudaDevAttrHostNativeAtomicSupported;
attr6 = attr5;
checkError(cudaDeviceGetAttribute(&val, attr5, dev_id));

// CHECK: /*
// CHECK-NEXT: DPCT1076:{{[0-9]+}}: The device attribute was not recognized. You may need to adjust the code.
// CHECK-NEXT: */
// CHECK_NEXT: cudaDeviceGetAttribute(&result, attr6, dev_id);
cudaDeviceGetAttribute(&val, attr6, dev_id);

int computeMode = -1, minor = 0;
// CHECK: /*
// CHECK-NEXT: DPCT1035:{{[0-9]+}}: All SYCL devices can be used by host to submit tasks. You may need to adjust this code.
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
// CHECK-NEXT:DPCT1007:{{[0-9]+}}: Migration of cudaDeviceGetP2PAttribute is not supported.
// CHECK-NEXT:*/
// CHECK-NEXT: accessSupported = 0;
cudaDeviceGetP2PAttribute(&accessSupported, cudaDevP2PAttrAccessSupported, device1, device2);

// CHECK:/*
// CHECK-NEXT:DPCT1007:{{[0-9]+}}: Migration of cudaDeviceGetP2PAttribute is not supported.
// CHECK-NEXT:*/
// CHECK-NEXT: perfRank = 0;
cudaDeviceGetP2PAttribute(&perfRank, cudaDevP2PAttrPerformanceRank, device1, device2);

// CHECK:/*
// CHECK-NEXT:DPCT1007:{{[0-9]+}}: Migration of cudaDeviceGetP2PAttribute is not supported.
// CHECK-NEXT:*/
// CHECK-NEXT: atomicSupported = 0;
cudaDeviceGetP2PAttribute(&atomicSupported, cudaDevP2PAttrNativeAtomicSupported, device1, device2);


char pciBusId[80];
// CHECK:/*
// CHECK-NEXT:DPCT1007:{{[0-9]+}}: Migration of cudaDeviceGetPCIBusId is not supported.
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
// CHECK-NEXT:DPCT1093:{{[0-9]+}}: The "device2" may not be the best XPU device. Adjust the selected device if needed.
// CHECK-NEXT:*/
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
// CHECK-NEXT:*/
// CHECK-NEXT:error_code = (dpct::select_device(device2), 0);
error_code = cudaSetDevice(device2);
// CHECK:/*
// CHECK-NEXT:DPCT1093:{{[0-9]+}}: The "device2" may not be the best XPU device. Adjust the selected device if needed.
// CHECK-NEXT:*/
// CHECK-NEXT: dpct::select_device(device2);
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
// CHECK-NEXT:DPCT1026:{{[0-9]+}}: The call to cudaPeekAtLastError was removed because this call is redundant in SYCL.
// CHECK-NEXT:*/
// CHECK-NEXT:dpct::get_current_device().queues_wait_and_throw();
int e1 = cudaPeekAtLastError();
cudaPeekAtLastError();
cudaThreadSynchronize();
return 0;
}

