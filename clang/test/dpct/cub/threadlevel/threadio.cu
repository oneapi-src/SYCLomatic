// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: c2s -in-root %S -out-root %T/threadlevel/threadio %S/threadio.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/threadlevel/threadio/threadio.dp.cpp --match-full-lines %s

#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

// CHECK: void ThreadLoadKernel(sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:     int *d_in;
// CHECK-NEXT:     int val = *(d_in + item_ct1.get_local_id(2));
// CHECK-NEXT:   }
__global__ void ThreadLoadKernel() {
  int *d_in;
  int val = cub::ThreadLoad<cub::LOAD_CA>(d_in + threadIdx.x);
}

// CHECK: void ThreadStoreKernel() {
// CHECK-NEXT:     int *d_out;
// CHECK-NEXT:     int val;
// CHECK-NEXT:     *(d_out) = val;
// CHECK-NEXT:   }  
__global__ void ThreadStoreKernel() {
  int *d_out;
  int val;
  cub::ThreadStore<cub::STORE_CG>(d_out, val);
}

int main(){
  ThreadLoadKernel<<<10, 10>>>();
  ThreadStoreKernel<<<10, 10>>>();
  return 0;
}
  