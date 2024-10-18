// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=kernel | FileCheck %s -check-prefix=KERNEL

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping="<<<>>>" | FileCheck %s -check-prefix=KERNEL

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping="<<<" | FileCheck %s -check-prefix=KERNEL

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=">>>" | FileCheck %s -check-prefix=KERNEL

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping="kernel<<<...>>>" | FileCheck %s -check-prefix=KERNEL

// KERNEL: CUDA API:
// KERNEL-NEXT:   f<<<gridDim, blockDim>>>();
// KERNEL-NEXT: Is migrated to:
// KERNEL-NEXT:   dpct::get_in_order_queue().parallel_for(
// KERNEL-NEXT:     sycl::nd_range<3>(gridDim * blockDim, blockDim),
// KERNEL-NEXT:     [=](sycl::nd_item<3> item_ct1) {
// KERNEL-NEXT:       f();
// KERNEL-NEXT:     });

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__constant__ | FileCheck %s -check-prefix=__CONSTANT__

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__constant | FileCheck %s -check-prefix=__CONSTANT__

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=constant | FileCheck %s -check-prefix=__CONSTANT__

// __CONSTANT__: CUDA API:
// __CONSTANT__-NEXT: __constant__ int v;
// __CONSTANT__-NEXT: Is migrated to:
// __CONSTANT__-NEXT: static dpct::constant_memory<int, 0> v;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__device__ | FileCheck %s -check-prefix=__DEVICE__

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__device | FileCheck %s -check-prefix=__DEVICE__

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=device | FileCheck %s -check-prefix=__DEVICE__

// __DEVICE__: CUDA API:
// __DEVICE__-NEXT: __device__ int v;
// __DEVICE__-NEXT: __device__ void f() {}
// __DEVICE__-NEXT: Is migrated to:
// __DEVICE__-NEXT: dpct::global_memory<int, 0> v;
// __DEVICE__-NEXT: void f() {}

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__global__ | FileCheck %s -check-prefix=__GLOBAL__

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__global | FileCheck %s -check-prefix=__GLOBAL__

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=global | FileCheck %s -check-prefix=__GLOBAL__

// __GLOBAL__: CUDA API:
// __GLOBAL__-NEXT: __global__ void f() {}
// __GLOBAL__-NEXT: Is migrated to:
// __GLOBAL__-NEXT: void f() {}

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__host__ | FileCheck %s -check-prefix=__HOST__

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__host | FileCheck %s -check-prefix=__HOST__

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=host | FileCheck %s -check-prefix=__HOST__

// __HOST__: CUDA API:
// __HOST__-NEXT: __host__ void f() {}
// __HOST__-NEXT: Is migrated to:
// __HOST__-NEXT: void f() {}

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__managed__ | FileCheck %s -check-prefix=__MANAGED__

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__managed | FileCheck %s -check-prefix=__MANAGED__

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=managed | FileCheck %s -check-prefix=__MANAGED__

// __MANAGED__: CUDA API:
// __MANAGED__-NEXT: __managed__ int v;
// __MANAGED__-NEXT: Is migrated to:
// __MANAGED__-NEXT: dpct::shared_memory<int, 0> v;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__shared__ | FileCheck %s -check-prefix=__SHARED__

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__shared | FileCheck %s -check-prefix=__SHARED__

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=shared | FileCheck %s -check-prefix=__SHARED__

// __SHARED__: CUDA API:
// __SHARED__-NEXT: __global__ void f() { __shared__ int v; }
// __SHARED__-NEXT: Is migrated to:
// __SHARED__-NEXT: void f(int &v) { }
