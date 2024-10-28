// UNSUPPORTED: system-windows

// RUN: cp %S/* .
// RUN: dpct --helper-function-preference=no-queue-device -p=%S -out-root %T --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --input-file %T/kernel1.dp.cpp --match-full-lines %S/kernel1.cu
// RUN: %if build_lit %{icpx -c -fsycl %T/kernel1.dp.cpp -o %T/kernel1.dp.o %}
// RUN: FileCheck --input-file %T/kernel2.dp.cpp --match-full-lines %S/kernel2.cu
// RUN: %if build_lit %{icpx -c -fsycl %T/kernel2.dp.cpp -o %T/kernel2.dp.o %}
// RUN: FileCheck --input-file %T/main.dp.cpp --match-full-lines %S/main.cu
// RUN: %if build_lit %{icpx -DNO_BUILD_TEST -c -fsycl %T/main.dp.cpp -o %T/main.dp.o %}

// RUN: grep -x "extern sycl::device dev_ct1;" %T/*.cpp | wc -l > %T/extern_count1.txt
// RUN: FileCheck --input-file %T/extern_count1.txt --match-full-lines extern_count.txt
// RUN: grep -x "extern sycl::queue q_ct1;" %T/*.cpp | wc -l > %T/extern_count2.txt
// RUN: FileCheck --input-file %T/extern_count2.txt --match-full-lines extern_count.txt
// RUN: grep -x "sycl::device dev_ct1(sycl::default_selector_v);" %T/*.cpp | wc -l > %T/define_count1.txt
// RUN: FileCheck --input-file %T/define_count1.txt --match-full-lines define_count.txt
// RUN: grep -x "sycl::queue q_ct1(dev_ct1, sycl::property_list{sycl::property::queue::in_order()});" %T/*.cpp | wc -l > %T/define_count2.txt
// RUN: FileCheck --input-file %T/define_count2.txt --match-full-lines define_count.txt

#include "common.cuh"
#include <stdio.h>

void f() {
  CUcontext ctx;
  CUdevice device;
  cudaStream_t *stream;
  int major;
  int minor;
  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cuCtxCreate_v2 was removed because it is redundant if it is migrated with option --helper-function-preference=no-queue-device which declares a global SYCL device and queue.
  // CHECK-NEXT: */
  // CHECK-NEXT: ctx = device;
  cuCtxCreate_v2(&ctx, 0, device);
  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cuDevicePrimaryCtxRetain was removed because it is redundant if it is migrated with option --helper-function-preference=no-queue-device which declares a global SYCL device and queue.
  // CHECK-NEXT: */
  // CHECK-NEXT: ctx = device;
  cuDevicePrimaryCtxRetain(&ctx, device);
  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cuCtxGetDevice was removed because it is redundant if it is migrated with option --helper-function-preference=no-queue-device which declares a global SYCL device and queue.
  // CHECK-NEXT: */
  // CHECK-NEXT: device = 0;
  cuCtxGetDevice(&device);
  // CHECK: device = dpct::get_major_version(dev_ct1);
  cudaDriverGetVersion(&device);
  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudaSetDevice was removed because it is redundant if it is migrated with option --helper-function-preference=no-queue-device which declares a global SYCL device and queue.
  // CHECK-NEXT: */
  // CHECK-NEXT: 0;
  cudaSetDevice(device);
  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cudaGetDevice was removed because it is redundant if it is migrated with option --helper-function-preference=no-queue-device which declares a global SYCL device and queue.
  // CHECK-NEXT: */
  // CHECK-NEXT: device = 0;
  cudaGetDevice(&device);
  // CHECK: major = dpct::get_major_version(dev_ct1);
  // CHECK-NEXT: minor = dpct::get_minor_version(dev_ct1);
  cuDeviceComputeCapability(&major, &minor, device);
  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cuCtxSetCurrent was removed because it is redundant if it is migrated with option --helper-function-preference=no-queue-device which declares a global SYCL device and queue.
  // CHECK-NEXT: */
  // CHECK-NEXT: 0;
  cuCtxSetCurrent(ctx);
  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cuCtxGetCurrent was removed because it is redundant if it is migrated with option --helper-function-preference=no-queue-device which declares a global SYCL device and queue.
  // CHECK-NEXT: */
  // CHECK-NEXT: ctx = 0;
  cuCtxGetCurrent(&ctx);
  // CHECK: *(stream) = new sycl::queue(dev_ct1, sycl::property_list{sycl::property::queue::in_order()});
  cudaStreamCreate(stream);
  // CHECK: *(stream) = new sycl::queue(dev_ct1, sycl::property_list{sycl::property::queue::in_order()});
  cudaStreamCreateWithFlags(stream, 1);
  // CHECK: *(stream) = new sycl::queue(dev_ct1, sycl::property_list{sycl::property::queue::in_order()});
  cudaStreamCreateWithPriority(stream, 1, 2);
}

// CHECK: int main() {
// CHECK-NEXT:   int *h_Data;
// CHECK-NEXT:   int *d_Data;
// CHECK-NEXT:   dpct::device_info deviceProp;
// CHECK-NEXT:   /*
// CHECK-NEXT:   DPCT1005:{{[0-9]+}}: The SYCL device version is different from CUDA Compute Compatibility. You may need to rewrite this code.
// CHECK-NEXT:   */
// CHECK-NEXT:   deviceProp.set_major_version(0);
// CHECK-NEXT: #ifndef NO_BUILD_TEST
// CHECK-NEXT:   dev_ct1.get_device_info(deviceProp);
// CHECK-NEXT: #endif
// CHECK-NEXT:   h_Data = (int *)malloc(SIZE * sizeof(int));
// CHECK-NEXT:   d_Data = sycl::malloc_device<int>(SIZE, q_ct1);
// CHECK-NEXT:   malloc1();
// CHECK-NEXT:   kernelWrapper1(d_Data);
// CHECK-NEXT:   q_ct1.wait_and_throw();
// CHECK-NEXT:   q_ct1.memcpy(h_Data, d_Data, SIZE * sizeof(int)).wait();
// CHECK-NEXT:   free1();
// CHECK-NEXT:   malloc2();
// CHECK-NEXT:   kernelWrapper2(d_Data);
// CHECK-NEXT:   q_ct1.wait_and_throw();
// CHECK-NEXT:   q_ct1.memcpy(h_Data, d_Data, SIZE * sizeof(int)).wait();
// CHECK-NEXT:   free2();
// CHECK-NEXT:   q_ct1.wait_and_throw();
// CHECK-NEXT:   sycl::free(d_Data, q_ct1);
// CHECK-NEXT:   free(h_Data);
// CHECK-NEXT:   printf("test pass!\n");
// CHECK-NEXT:   return 0;
// CHECK-NEXT: }
int main() {
  int *h_Data;
  int *d_Data;
  cudaDeviceProp deviceProp;
  deviceProp.major = 0;
#ifndef NO_BUILD_TEST
  cudaGetDeviceProperties(&deviceProp, 0);
#endif
  h_Data = (int *)malloc(SIZE * sizeof(int));
  cudaMalloc((void **)&d_Data, SIZE * sizeof(int));
  malloc1();
  kernelWrapper1(d_Data);
  cudaDeviceSynchronize();
  cudaMemcpy(h_Data, d_Data, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
  free1();
  malloc2();
  kernelWrapper2(d_Data);
  cudaDeviceSynchronize();
  cudaMemcpy(h_Data, d_Data, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
  free2();
  cudaFree(d_Data);
  free(h_Data);
  printf("test pass!\n");
  return 0;
}
