// RUN: cu2sycl -out-root %T %s -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/device001.sycl.cpp

int main(int argc, char **argv)
{
int deviceCount = 0;
// CHECK: deviceCount = cu2sycl::get_device_manager().device_count();
cudaGetDeviceCount(&deviceCount);

int dev_id;
// CHECK: dev_id = cu2sycl::get_device_manager().current_device_id();
cudaGetDevice(&dev_id);

cudaDeviceProp deviceProp;
// CHECK: deviceProp = cu2sycl::get_device_manager().get_device( 0).get_device_info();
cudaGetDeviceProperties(&deviceProp, 0);

int atomicSupported;
// CHECK: atomicSupported = cu2sycl::get_device_manager().get_device(  dev_id).is_native_atomic_supported();
cudaDeviceGetAttribute(&atomicSupported, cudaDevAttrHostNativeAtomicSupported, dev_id);

int device1 = 0;
int device2 = 1;
int perfRank = 0;
int accessSupported = 0;
// CHECK: accessSupported = 0;
cudaDeviceGetP2PAttribute(&accessSupported, cudaDevP2PAttrAccessSupported, device1, device2);
// CHECK: perfRank = 0;
cudaDeviceGetP2PAttribute(&perfRank, cudaDevP2PAttrPerformanceRank, device1, device2);
// CHECK: atomicSupported = 0;
cudaDeviceGetP2PAttribute(&atomicSupported, cudaDevP2PAttrNativeAtomicSupported, device1, device2);

// CHECK: ;
cudaDeviceReset();
// CHECK: cu2sycl::get_device_manager().select_device(device2);
cudaSetDevice(device2);

// CHECK: int e = 0;
int e = cudaGetLastError();

return 0;
}
