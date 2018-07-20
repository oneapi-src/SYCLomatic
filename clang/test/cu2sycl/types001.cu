// RUN: cu2sycl -out-root %T %s -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/types001.sycl.cpp

// CHECK: cu2sycl::sycl_device_info deviceProp;
cudaDeviceProp deviceProp;

// CHECK: const cu2sycl::sycl_device_info deviceProp1 = {};
const cudaDeviceProp deviceProp1 = {};

// CHECK: volatile cu2sycl::sycl_device_info deviceProp2;
volatile cudaDeviceProp deviceProp2;

// CHECK:  void foo(cu2sycl::sycl_device_info p) {
void foo(cudaDeviceProp p) {
    return;
}
