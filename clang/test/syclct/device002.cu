// RUN: syclct -out-root %T %s -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck %s --match-full-lines --input-file %T/device002.sycl.cpp

#include <stdio.h>

void checkError(cudaError_t err) {

}

int main(int argc, char **argv)
{
int devID = atoi(argv[1]);
cudaDeviceProp cdp;
//CHECK: int error_code = syclct::get_device_manager().get_device( devID).get_device_info(cdp);
cudaError_t error_code = cudaGetDeviceProperties(&cdp, devID);

if (error_code == cudaSuccess) {
// CHECK: /*
// CHECK-NEXT:  SYCLCT1005:3: The device version is different. You may want to rewrite this code
// CHECK-NEXT: */
// CHECK-NEXT: /*
// CHECK-NEXT:  SYCLCT1006:4: SYCL doesn't provide standard API to differentiate between integrated/discrete GPU devices. Consider to re-implement the code which depends on this field
// CHECK-NEXT: */
    if (cdp.major < 3 && cdp.integrated != 1) {
            printf("do_complex_compute requires compute capability 3.0 or later and not integrated\n");
    }
}

int deviceCount = 0;
// CHECK: deviceCount = syclct::get_device_manager().device_count();
cudaGetDeviceCount(&deviceCount);

int dev_id;
// CHECK: dev_id = syclct::get_device_manager().current_device_id();
cudaGetDevice(&dev_id);

cudaDeviceProp deviceProp;
// CHECK: syclct::get_device_manager().get_device( 0).get_device_info(deviceProp);
cudaGetDeviceProperties(&deviceProp, 0);

int atomicSupported;
// CHECK: atomicSupported = syclct::get_device_manager().get_device(  dev_id).is_native_atomic_supported();
cudaDeviceGetAttribute(&atomicSupported, cudaDevAttrHostNativeAtomicSupported, dev_id);

int device1 = 0;
int device2 = 1;
int perfRank = 0;
int accessSupported = 0;

// CHECK:/*
// CHECK-NEXT:SYCLCT1004:{{[0-9]+}}: P2P Access is not supported in Sycl
// CHECK-NEXT:*/
// CHECK-NEXT: accessSupported = 0;
cudaDeviceGetP2PAttribute(&accessSupported, cudaDevP2PAttrAccessSupported, device1, device2);

// CHECK:/*
// CHECK-NEXT:SYCLCT1004:{{[0-9]+}}: P2P Access is not supported in Sycl
// CHECK-NEXT:*/
// CHECK-NEXT: perfRank = 0;
cudaDeviceGetP2PAttribute(&perfRank, cudaDevP2PAttrPerformanceRank, device1, device2);

// CHECK:/*
// CHECK-NEXT:SYCLCT1004:{{[0-9]+}}: P2P Access is not supported in Sycl
// CHECK-NEXT:*/
// CHECK-NEXT: atomicSupported = 0;
cudaDeviceGetP2PAttribute(&atomicSupported, cudaDevP2PAttrNativeAtomicSupported, device1, device2);

// CHECK: syclct::get_device_manager().current_device().reset();
cudaDeviceReset();
// CHECK: syclct::get_device_manager().select_device(device2);
cudaSetDevice(device2);

// CHECK:syclct::get_device_manager().current_device().queues_wait_and_throw();
// CHECK-NEXT:/*
// CHECK-NEXT:SYCLCT1003:{{[0-9]+}}: Translated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
// CHECK-NEXT:*/
// CHECK-NEXT:int err = (syclct::get_device_manager().current_device().queues_wait_and_throw(), 0);
// CHECK-NEXT:/*
// CHECK-NEXT:SYCLCT1003:{{[0-9]+}}: Translated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
// CHECK-NEXT:*/
// CHECK-NEXT:checkError((syclct::get_device_manager().current_device().queues_wait_and_throw(), 0));
// CHECK-NEXT:/*
// CHECK-NEXT:SYCLCT1003:{{[0-9]+}}: Translated api does not return error code. (*, 0) is inserted. You may want to rewrite this code
// CHECK-NEXT:*/
// CHECK-NEXT:return (syclct::get_device_manager().current_device().queues_wait_and_throw(), 0);
cudaDeviceSynchronize();
cudaError_t err = cudaDeviceSynchronize();
checkError(cudaDeviceSynchronize());
return cudaDeviceSynchronize();
// CHECK: int e = 0;
int e = cudaGetLastError();
// CHECK:syclct::get_device_manager().current_device().queues_wait_and_throw();
cudaThreadSynchronize();
return 0;
}
