// RUN: dpct --format-range=none --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -std=c++14  -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/sharedmem-dim-noUSM.dp.cpp
// CHECK: #define DPCT_USM_LEVEL_NONE
#include <cuda_runtime.h>

// CHECK: /*
// CHECK-NEXT: DPCT1060:{{[0-9]+}}: SYCL range can only be a 1D, 2D or 3D vector. Adjust the code.
// CHECK-NEXT: */
// CHECK-NEXT: dpct::global_memory<int, 4> dev_mem(10, 10, 10, 10);
__device__ int dev_mem[10][10][10][10];

__shared__ int sha1_mem[10][10][10][10];

// CHECK: /*
// CHECK-NEXT: DPCT1060:{{[0-9]+}}: SYCL range can only be a 1D, 2D or 3D vector. Adjust the code.
// CHECK-NEXT: */
// CHECK-NEXT: dpct::shared_memory<int, 4> man_mem(10, 10, 10, 10);
__managed__ int man_mem[10][10][10][10];

// CHECK: /*
// CHECK-NEXT: DPCT1060:{{[0-9]+}}: SYCL range can only be a 1D, 2D or 3D vector. Adjust the code.
// CHECK-NEXT: */
// CHECK-NEXT: dpct::constant_memory<int, 4> con_mem(10, 10, 10, 10);
__constant__ int con_mem[10][10][10][10];

__global__ void staticReverse()
{
  __shared__ int sha2_mem[10][10][10][10];

}

__global__ void dynamicReverse()
{
  extern __shared__ int shad_mem[][2][2][2];

  int p = shad_mem[0][0][0][0];
}

int main(void)
{
// CHECK: /*
// CHECK-NEXT: DPCT1060:{{[0-9]+}}: SYCL range can only be a 1D, 2D or 3D vector. Adjust the code.
// CHECK-NEXT: */
// CHECK-NEXT: sycl::accessor<int, 4, sycl::access::mode::read_write, sycl::access::target::local> sha2_mem_acc_ct1(sha2_mem_range_ct1, cgh);

  staticReverse<<<10,10>>>();

// CHECK: /*
// CHECK-NEXT: DPCT1060:{{[0-9]+}}: SYCL range can only be a 1D, 2D or 3D vector. Adjust the code.
// CHECK-NEXT: */
// CHECK-NEXT: sycl::accessor<uint8_t, 4, sycl::access::mode::read_write, sycl::access::target::local> dpct_local_acc_ct1(dpct_local_range_ct1, cgh);

  dynamicReverse<<<10,10,16*sizeof(int)>>>();
  return 0;
}