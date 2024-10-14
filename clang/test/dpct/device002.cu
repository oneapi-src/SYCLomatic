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
// CHECK: dpct::err0 error_code = DPCT_CHECK_ERROR(dpct::get_device(devID).get_device_info(cdp));
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
// CHECK: deviceCount = dpct::device_count();
cudaGetDeviceCount(&deviceCount);

int dev_id;
// CHECK: dev_id = dpct::get_current_device_id();
cudaGetDevice(&dev_id);

cudaDeviceProp deviceProp;
// CHECK: dpct::get_device(0).get_device_info(deviceProp);
cudaGetDeviceProperties(&deviceProp, 0);

int atomicSupported;
// CHECK: atomicSupported = dpct::get_device(dev_id).is_native_atomic_supported();
cudaDeviceGetAttribute(&atomicSupported, cudaDevAttrHostNativeAtomicSupported, dev_id);

int val;
// CHECK: val = dpct::get_device(dev_id).get_major_version();
cudaDeviceGetAttribute(&val, cudaDevAttrComputeCapabilityMajor, dev_id);

struct attr{
    cudaDeviceAttr attr;
} attr1;
// CHECK: /*
// CHECK-NEXT: DPCT1076:{{[0-9]+}}: The device attribute was not recognized. You may need to adjust the code.
// CHECK-NEXT: */
// CHECK-NEXT: cudaDeviceGetAttribute(&val, attr1.attr, dev_id);
cudaDeviceGetAttribute(&val, attr1.attr, dev_id);

// CHECK: int attr2 = 86;
// CHECK-NEXT: atomicSupported = dpct::get_device(dev_id).is_native_atomic_supported();
cudaDeviceAttr attr2 = cudaDevAttrHostNativeAtomicSupported;
cudaDeviceGetAttribute(&atomicSupported, attr2, dev_id);

// CHECK: int attr3;
// CHECK-NEXT: attr3 = 75;
// CHECK-NEXT: val = dpct::get_device(dev_id).get_major_version();
cudaDeviceAttr attr3;
attr3 = cudaDevAttrComputeCapabilityMajor;
cudaDeviceGetAttribute(&val, attr3, dev_id);

// CHECK: int attr4;
// CHECK-NEXT: attr4 = 86;
// CHECK-NEXT: attr4 = 75;
// CHECK-NEXT: val = dpct::get_device(dev_id).get_major_version();
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
// CHECK-NEXT: cudaDeviceGetAttribute(&val, attr5, dev_id);
cudaDeviceAttr attr5;
int somecondition;
attr5 = cudaDevAttrHostNativeAtomicSupported;
attr5 = cudaDevAttrComputeCapabilityMajor;
if(somecondition)
  attr5 = cudaDevAttrHostNativeAtomicSupported;
cudaDeviceGetAttribute(&val, attr5, dev_id);

// CHECK: attr5 = 86;
// CHECK-NEXT: attr6 = attr5;
// CHECK-NEXT: checkError(DPCT_CHECK_ERROR(val = dpct::get_device(dev_id).is_native_atomic_supported()));
attr5 = cudaDevAttrHostNativeAtomicSupported;
attr6 = attr5;
checkError(cudaDeviceGetAttribute(&val, attr5, dev_id));

// CHECK: /*
// CHECK-NEXT: DPCT1076:{{[0-9]+}}: The device attribute was not recognized. You may need to adjust the code.
// CHECK-NEXT: */
// CHECK-NEXT: cudaDeviceGetAttribute(&val, attr6, dev_id);
cudaDeviceGetAttribute(&val, attr6, dev_id);

int computeMode = -1, minor = 0;
// CHECK: /*
// CHECK-NEXT: DPCT1035:{{[0-9]+}}: All SYCL devices can be used by the host to submit tasks. You may need to adjust this code.
// CHECK-NEXT: */
// CHECK-NEXT: checkError(DPCT_CHECK_ERROR(computeMode = 1));
checkError(cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, dev_id));
// CHECK: checkError(DPCT_CHECK_ERROR(minor = dpct::get_device(dev_id).get_minor_version()));
checkError(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev_id));

int multiProcessorCount = 0, clockRate = 0;
// CHECK: checkError(DPCT_CHECK_ERROR(multiProcessorCount = dpct::get_device(dev_id).get_max_compute_units()));
checkError(cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, dev_id));
// CHECK: checkError(DPCT_CHECK_ERROR(clockRate = dpct::get_device(dev_id).get_max_clock_frequency()));
checkError(cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, dev_id));

int integrated = -1;
// CHECK: checkError(DPCT_CHECK_ERROR(integrated = dpct::get_device(dev_id).get_integrated()));
checkError(cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, dev_id));

int alignment;
// CHECK: /*
// CHECK-NEXT: DPCT1051:{{[0-9]+}}: SYCL does not support a device property functionally compatible with cudaDevAttrTextureAlignment. It was migrated to get_mem_base_addr_align_in_bytes. You may need to adjust the value of get_mem_base_addr_align_in_bytes for the specific device.
// CHECK-NEXT: */
// CHECK-NEXT: checkError(DPCT_CHECK_ERROR(alignment = dpct::get_device(dev_id).get_mem_base_addr_align_in_bytes()));
checkError(cudaDeviceGetAttribute(&alignment, cudaDevAttrTextureAlignment, dev_id));

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

// CHECK:error_code = DPCT_CHECK_ERROR(dpct::get_current_device().reset());
error_code = cudaDeviceReset();

// CHECK: dpct::get_current_device().reset();
cudaThreadExit();

// CHECK:error_code = DPCT_CHECK_ERROR(dpct::get_current_device().reset());
error_code = cudaThreadExit();

// CHECK:error_code = DPCT_CHECK_ERROR(dpct::select_device(device2));
error_code = cudaSetDevice(device2);
// CHECK:/*
// CHECK-NEXT:DPCT1093:{{[0-9]+}}: The "device2" device may be not the one intended for use. Adjust the selected device if needed.
// CHECK-NEXT:*/
// CHECK-NEXT: dpct::select_device(device2);
cudaSetDevice(device2);

// CHECK:dpct::get_current_device().queues_wait_and_throw();
// CHECK-NEXT:dpct::err0 err = DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw());
// CHECK-NEXT:checkError(DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()));
// CHECK-NEXT:return DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw());
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
// CHECK-NEXT:DPCT1026:{{[0-9]+}}: The call to cudaPeekAtLastError was removed because this functionality is redundant in SYCL.
// CHECK-NEXT:*/
// CHECK-NEXT:dpct::get_current_device().queues_wait_and_throw();
int e1 = cudaPeekAtLastError();
cudaPeekAtLastError();
cudaThreadSynchronize();
return 0;
}

