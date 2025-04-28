// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct  --format-range=none -out-root %T/cvta %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only 
// RUN: FileCheck %s --match-full-lines --input-file %T/cvta/cvta.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/cvta/cvta.dp.cpp -o %T/cvta/cvta.dp.o %}

// clang-format off
#include <cstdint>
#include <cuda_runtime.h>


// CHECK: void test_cvta_to_shared_u64(uint64_t* output, int *shared_data) {
// CHECK-NEXT:     // Shared memory 
// CHECK-NEXT:    shared_data[0] = 0;
// CHECK-NEXT:    uint64_t shared_addr = 0;
// CHECK-NEXT:    shared_addr = (uint64_t)(&shared_data[0]);
// CHECK-NEXT:    output[sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2)] = shared_addr;
// CHECK-NEXT:}
__global__ void test_cvta_to_shared_u64(uint64_t* output) {
    __shared__ int shared_data[1]; // Shared memory
    shared_data[0] = 0;
    uint64_t shared_addr = 0;
    asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(shared_addr) : "l"(&shared_data[0]));
    output[threadIdx.x] = shared_addr;
}


#define N 128
// CHECK: void testKernel(unsigned int *addr_out, int *B_shared) {
// CHECK-NEXT:     auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
// CHECK-NEXT:      // Shared memory
// CHECK-NEXT:     unsigned int addr1;
// CHECK-NEXT:     int k_0_1 = item_ct1.get_group(2);
// CHECK-NEXT:     int ax1_0 = item_ct1.get_local_id(2);
// CHECK-NEXT:     {
// CHECK-NEXT:         uint64_t addr;
// CHECK-NEXT:         addr = (uint64_t)((void *)((&(B_shared[(((k_0_1 * (N * 16 + 128)) + (((int)item_ct1.get_local_id(1)) * (N / 2))) + (ax1_0 * 16))])) +
// CHECK-NEXT:                        (((((int)item_ct1.get_local_id(2)) & 15) * (N + 8)) + ((((int)item_ct1.get_local_id(2)) >> 4) * 8))));
// CHECK-NEXT:         addr1 = static_cast<uint32_t>(addr);
// CHECK-NEXT:     }
// CHECK-NEXT:     addr_out[item_ct1.get_local_id(2)] = addr1;
// CHECK-NEXT: }
__global__ void testKernel(unsigned int *addr_out) {
    __shared__ int B_shared[N * 16 + 128]; // Shared memory
    unsigned int addr1;
    int k_0_1 = blockIdx.x;
    int ax1_0 = threadIdx.x;
    __asm__ __volatile__(
        "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
        : "=r"(addr1)
        : "l"((void *)((&(B_shared[(((k_0_1 * (N * 16 + 128)) + (((int)threadIdx.y) * (N / 2))) + (ax1_0 * 16))])) +
                       (((((int)threadIdx.x) & 15) * (N + 8)) + ((((int)threadIdx.x) >> 4) * 8)))));
    addr_out[threadIdx.x] = addr1;
}


// CHECK: void read_shared_value(int *output, int *shared_data) {
// CHECK-NEXT:  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
// CHECK-NEXT:   // Shared memory allocation
// CHECK-NEXT:  if (item_ct1.get_local_id(2) == 0) {
// CHECK-NEXT:    shared_data[0] = 42;
// CHECK-NEXT:  }
// CHECK-NEXT:  item_ct1.barrier(sycl::access::fence_space::local_space);
// CHECK-NEXT:  unsigned long long shared_addr_u64;
// CHECK-NEXT:  int value;
// CHECK-NEXT:  shared_addr_u64 = (uint64_t)(shared_data);
// CHECK-NEXT:  value = *((uint32_t *)(uintptr_t)shared_addr_u64);
// CHECK-NEXT:  if (item_ct1.get_local_id(2) == 0) {
// CHECK-NEXT:    output[0] = value;
// CHECK-NEXT:  }
// CHECK-NEXT:}
__global__ void read_shared_value(int *output) {
  __shared__ int shared_data[1]; // Shared memory allocation
  if (threadIdx.x == 0) {
    shared_data[0] = 42;
  }
  __syncthreads();
  unsigned long long shared_addr_u64;
  int value;
  asm volatile(
      "cvta.to.shared.u64 %0, %2;\n\t" // Properly uses input operand %2
      "ld.shared.u32 %1, [%0];\n\t"    // Correctly assigns to output %1
      : "=l"(shared_addr_u64), "=r"(value)
      : "l"(shared_data));
  if (threadIdx.x == 0) {
    output[0] = value;
  }
}

// clang-format on
