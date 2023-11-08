// RUN: dpct --format-range=none -out-root %T/sharedmem-dim %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/sharedmem-dim/sharedmem-dim.dp.cpp
// CHECK: #include <sycl/sycl.hpp>
// CHECK: #include <dpct/dpct.hpp>

#include <cuda_runtime.h>


// CHECK-NOT: DPCT1060:{{[0-9]+}}: SYCL range can only be a 1D, 2D, or 3D vector. Adjust the code.

// CHECK: dpct::global_memory<int, 4> dev_mem(10, 10, 10, 10);
__device__ int dev_mem[10][10][10][10];

__shared__ int sha1_mem[10][10][10][10];


// CHECK-NOT: DPCT1060:{{[0-9]+}}: SYCL range can only be a 1D, 2D, or 3D vector. Adjust the code.

// CHECK: dpct::shared_memory<int, 4> man_mem(10, 10, 10, 10);
__managed__ int man_mem[10][10][10][10];


// CHECK-NOT: DPCT1060:{{[0-9]+}}: SYCL range can only be a 1D, 2D, or 3D vector. Adjust the code.

// CHECK: static dpct::constant_memory<int, 4> con_mem(10, 10, 10, 10);
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
// CHECK-NEXT: DPCT1060:{{[0-9]+}}: SYCL range can only be a 1D, 2D, or 3D vector. Adjust the code.
// CHECK-NEXT: */
// CHECK-NEXT: sycl::local_accessor<int, 4> sha2_mem_acc_ct1(sycl::range<4>(10, 10, 10, 10), cgh);

  staticReverse<<<10,10>>>();

// CHECK: /*
// CHECK-NEXT: DPCT1083:{{[0-9]+}}: The size of local memory in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK-NEXT: */
// CHECK: /*
// CHECK-NEXT: DPCT1060:{{[0-9]+}}: SYCL range can only be a 1D, 2D, or 3D vector. Adjust the code.
// CHECK-NEXT: */
// CHECK-NEXT: sycl::local_accessor<uint8_t, 4> dpct_local_acc_ct1(sycl::range<4>(16*sizeof(int), 1, 1, 1), cgh);

  dynamicReverse<<<10,10,16*sizeof(int)>>>();
  return 0;
}

void foo1() {
// CHECK: /*
// CHECK-NEXT: DPCT1083:{{[0-9]+}}: The size of local memory in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK-NEXT: */
// CHECK-NEXT: int shared_memory_size = 3 * sizeof(float);
  int shared_memory_size = 3 * sizeof(float);
  dynamicReverse<<<1, 1, shared_memory_size>>>();
}

void foo2() {
// CHECK: /*
// CHECK-NEXT: DPCT1083:{{[0-9]+}}: The size of local memory in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK-NEXT: */
// CHECK-NEXT: int bbb = sizeof(float);
  int aaa = 3;
  int bbb = sizeof(float);
  int shared_memory_size;
  shared_memory_size = aaa * bbb;
  dynamicReverse<<<1, 1, shared_memory_size>>>();
}

// CHECK: void kernel3(int *a) {
// CHECK-NOT: DPCT1083:{{[0-9]+}}: The size of local memory in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK: }
__global__ void kernel3() {
  __shared__ int a[sizeof(float3) * 3];
}

// CHECK: void foo3() {
// CHECK: dpct::get_in_order_queue().submit(
// CHECK-NEXT: [&](sycl::handler &cgh) {
// CHECK: /*
// CHECK-NEXT: DPCT1083:{{[0-9]+}}: The size of local memory in the migrated code may be different from the original code. Check that the allocated memory size in the migrated code is correct.
// CHECK-NEXT: */
// CHECK-NEXT: sycl::local_accessor<int, 1> a_acc_ct1(sycl::range<1>(sizeof(sycl::float3) * 3), cgh);
// CHECK: });
// CHECK: }
void foo3() {
  kernel3<<<1, 1>>>();
}
