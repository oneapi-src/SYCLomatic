// UNSUPPORTED: system-windows

// RUN: cp %S/* .
// RUN: dpct --use-pure-sycl-queue -p=%S -out-root %T --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --input-file %T/kernel1.dp.cpp --match-full-lines %S/kernel1.cu
// RUN: FileCheck --input-file %T/kernel2.dp.cpp --match-full-lines %S/kernel2.cu
// RUN: FileCheck --input-file %T/main.dp.cpp --match-full-lines %S/main.cu

// RUN: grep -x "extern sycl::device dev_ct1;" %T/*.cpp | wc -l > %T/extern_count1.txt
// RUN: FileCheck --input-file %T/extern_count1.txt --match-full-lines extern_count.txt
// RUN: grep -x "extern sycl::queue q_ct1;" %T/*.cpp | wc -l > %T/extern_count2.txt
// RUN: FileCheck --input-file %T/extern_count2.txt --match-full-lines extern_count.txt
// RUN: grep -x "sycl::device dev_ct1;" %T/*.cpp | wc -l > %T/define_count1.txt
// RUN: FileCheck --input-file %T/define_count1.txt --match-full-lines define_count.txt
// RUN: grep -x "sycl::queue q_ct1(dev_ct1, sycl::property_list{sycl::property::queue::in_order()});" %T/*.cpp | wc -l > %T/define_count2.txt
// RUN: FileCheck --input-file %T/define_count2.txt --match-full-lines define_count.txt

#include "common.cuh"
#include <stdio.h>

// CHECK: int main() {
// CHECK-NEXT:   int *h_Data;
// CHECK-NEXT:   int *d_Data;
// CHECK-NEXT:   dpct::device_info deviceProp;
// CHECK-NEXT:   /*
// CHECK-NEXT:   DPCT1005:0: The SYCL device version is different from CUDA Compute Compatibility. You may need to rewrite this code.
// CHECK-NEXT:   */
// CHECK-NEXT:   deviceProp.set_major_version(0);
// CHECK-NEXT:   dpct::get_device_info(deviceProp, dev_ct1);
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
  cudaGetDeviceProperties(&deviceProp, 0);
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
