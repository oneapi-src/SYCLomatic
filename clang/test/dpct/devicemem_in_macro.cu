// RUN: dpct --format-range=none -out-root %T/devicemem_in_macro %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/devicemem_in_macro/devicemem_in_macro.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/devicemem_in_macro/devicemem_in_macro.dp.cpp -o %T/devicemem_in_macro/devicemem_in_macro.dp.o %}

#include <cuda_runtime.h>

// CHECK: #define TABLE_BEGIN(type, name, size) static const type name[size] = {
// CHECK: #define TABLE_END() };
// CHECK: static dpct::global_memory<const uint8_t, 1> mem(sycl::range<1>(8), {1, 2, 4, 8, 16, 32, 64, 128});
// CHECK: static dpct::global_memory<const int, 1> a(sycl::range<1>(1), {1});
#define TABLE_BEGIN(type, name, size) static const __device__ type name[size] = {
#define TABLE_END() };
TABLE_BEGIN(uint8_t, mem, 8)
    1, 2, 4, 8, 16, 32, 64, 128
TABLE_END()

static const __device__ int a[1] = {1};

__global__ void kernel() {
  a[0];
  mem[1];
}


int main() {
  kernel<<<1, 1>>>();
  return 0;
}
