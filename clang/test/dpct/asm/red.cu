// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/red %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/red/red.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/red/red.dp.cpp -o %T/red/red.dp.o %}

// clang-format off
#include <cstdint>
#include <cuda_runtime.h>

// CHECK: void atomicAddKernel(int* lock, int val, const sycl::nd_item<3> &item_ct1) {
// CHECK-NEXT:     *lock = sycl::reduce_over_group(item_ct1.get_group(), val,sycl::plus<>());
// CHECK-NEXT:}
__global__ void atomicAddKernel(int* lock, int val) {
    asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n"
                 ::"l"(lock),"r"(val):"memory");
}

// CHECK: void atomicOrKernel(uint32_t* lock, uint32_t val,
// CHECK-NEXT:                     const sycl::nd_item<3> &item_ct1) {
// CHECK-NEXT:     *lock = sycl::reduce_over_group(item_ct1.get_group(), val,sycl::bit_or<>());
// CHECK-NEXT:}
__global__ void atomicOrKernel(uint32_t* lock, uint32_t val) {
    asm volatile("red.relaxed.gpu.global.or.b32 [%0], %1;\n"
                 ::"l"(lock),"r"(val):"memory");
}

// CHECK: void atomicXorKernel(uint32_t* lock, uint32_t val,
// CHECK-NEXT:                      const sycl::nd_item<3> &item_ct1) {
// CHECK-NEXT:     *lock = sycl::reduce_over_group(item_ct1.get_group(), val,sycl::bit_xor<>());
// CHECK-NEXT:}
__global__ void atomicXorKernel(uint32_t* lock, uint32_t val) {
    asm volatile("red.relaxed.gpu.global.xor.b32 [%0], %1;\n"
                 ::"l"(lock),"r"(val):"memory");
}

// CHECK: void atomicAndKernel(uint32_t* lock, uint32_t val,
// CHECK-NEXT:                     const sycl::nd_item<3> &item_ct1) {
// CHECK-NEXT:     *lock = sycl::reduce_over_group(item_ct1.get_group(), val,sycl::bit_and<>());
// CHECK-NEXT: }
__global__ void atomicAndKernel(uint32_t* lock, uint32_t val) {
    asm volatile("red.relaxed.gpu.global.and.b32 [%0], %1;\n"
                 ::"l"(lock),"r"(val):"memory");
}

// CHECK: void atomicMaxKernel(uint32_t* lock, uint32_t val,
// CHECK-NEXT:                      const sycl::nd_item<3> &item_ct1) {
// CHECK-NEXT:     *lock = sycl::reduce_over_group(item_ct1.get_group(), val,sycl::maximum<>());
// CHECK-NEXT: }
__global__ void atomicMaxKernel(uint32_t* lock, uint32_t val) {
    asm volatile("red.relaxed.gpu.global.max.u32 [%0], %1;\n"
                 ::"l"(lock),"r"(val):"memory");
}

// CHECK: void atomicMinKernel(uint32_t* lock, uint32_t val,
// CHECK-NEXT:                      const sycl::nd_item<3> &item_ct1) {
// CHECK-NEXT:     *lock = sycl::reduce_over_group(item_ct1.get_group(), val,sycl::minimum<>());
// CHECK-NEXT: }
__global__ void atomicMinKernel(uint32_t* lock, uint32_t val) {
    asm volatile("red.relaxed.gpu.global.min.u32 [%0], %1;\n"
                 ::"l"(lock),"r"(val):"memory");
}

// CHECK: void atomicAddKernelRelease(int* lock, int val,
// CHECK-NEXT:                        const sycl::nd_item<3> &item_ct1) {
// CHECK-NEXT:     *lock = sycl::reduce_over_group(item_ct1.get_group(), val,sycl::plus<>());
// CHECK-NEXT:}
__global__ void atomicAddKernelRelease(int* lock, int val) {
    asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n"
                 ::"l"(lock),"r"(val):"memory");
}

// CHECK: void atomicOrKernelRelease(uint32_t* lock, uint32_t val,
// CHECK-NEXT:                     const sycl::nd_item<3> &item_ct1) {
// CHECK-NEXT:     *lock = sycl::reduce_over_group(item_ct1.get_group(), val,sycl::bit_or<>());
// CHECK-NEXT:}
__global__ void atomicOrKernelRelease(uint32_t* lock, uint32_t val) {
    asm volatile("red.release.gpu.global.or.b32 [%0], %1;\n"
                 ::"l"(lock),"r"(val):"memory");
}

// CHECK: void atomicXorKernelRelease(uint32_t* lock, uint32_t val,
// CHECK-NEXT:                      const sycl::nd_item<3> &item_ct1) {
// CHECK-NEXT:     *lock = sycl::reduce_over_group(item_ct1.get_group(), val,sycl::bit_xor<>());
// CHECK-NEXT:}
__global__ void atomicXorKernelRelease(uint32_t* lock, uint32_t val) {
    asm volatile("red.release.gpu.global.xor.b32 [%0], %1;\n"
                 ::"l"(lock),"r"(val):"memory");
}

// CHECK: void atomicAndKernelRelease(uint32_t* lock, uint32_t val,
// CHECK-NEXT:                     const sycl::nd_item<3> &item_ct1) {
// CHECK-NEXT:     *lock = sycl::reduce_over_group(item_ct1.get_group(), val,sycl::bit_and<>());
// CHECK-NEXT: }
__global__ void atomicAndKernelRelease(uint32_t* lock, uint32_t val) {
    asm volatile("red.release.gpu.global.and.b32 [%0], %1;\n"
                 ::"l"(lock),"r"(val):"memory");
}

// CHECK: void atomicMaxKernelRelease(uint32_t* lock, uint32_t val,
// CHECK-NEXT:                      const sycl::nd_item<3> &item_ct1) {
// CHECK-NEXT:     *lock = sycl::reduce_over_group(item_ct1.get_group(), val,sycl::maximum<>());
// CHECK-NEXT: }
__global__ void atomicMaxKernelRelease(uint32_t* lock, uint32_t val) {
    asm volatile("red.release.gpu.global.max.u32 [%0], %1;\n"
                 ::"l"(lock),"r"(val):"memory");
}

// CHECK: void atomicMinKernelRelease(uint32_t* lock, uint32_t val,
// CHECK-NEXT:                      const sycl::nd_item<3> &item_ct1) {
// CHECK-NEXT:     *lock = sycl::reduce_over_group(item_ct1.get_group(), val,sycl::minimum<>());
// CHECK-NEXT: }
__global__ void atomicMinKernelRelease(uint32_t* lock, uint32_t val) {
    asm volatile("red.release.gpu.global.min.u32 [%0], %1;\n"
                 ::"l"(lock),"r"(val):"memory");
}
// clang-format on
