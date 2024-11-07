// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/shfl %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/shfl/shfl.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/shfl/shfl.dp.cpp -o %T/shfl/shfl.dp.o %}

// clang-format off
#include <cuda_runtime.h>
#include <cstdint>

__global__ void shfl() {
    int value; 
    unsigned mask = 0xFFFFFFFF;
    int offset;
    int output;                                              

    // CHECK: output = dpct::shift_sub_group_right(item_ct1.get_sub_group(), value, offset);
    asm volatile("shfl.up.b32 %0, %1, %2, %3;" : "=r"(output) : "r"(value), "r"(offset),"r"(mask));

    // CHECK: output = dpct::shift_sub_group_right(item_ct1.get_sub_group(), value, offset);
    asm volatile("shfl.up.b32 %0, %1, %2, 0xFFFFFFFF;" : "=r"(output) : "r"(value), "r"(offset));

    // CHECK: output = dpct::shift_sub_group_left(item_ct1.get_sub_group(), value, offset);
    asm volatile("shfl.down.b32 %0, %1, %2, %3;" : "=r"(output) : "r"(value), "r"(offset),"r"(mask));

    // CHECK: output = dpct::shift_sub_group_left(item_ct1.get_sub_group(), value, offset);
    asm volatile("shfl.down.b32 %0, %1, %2, 0xFFFFFFFF;" : "=r"(output) : "r"(value), "r"(offset));

    // CHECK: output = dpct::select_from_sub_group(item_ct1.get_sub_group(), value, offset);
    asm volatile("shfl.idx.b32 %0, %1, %2, %3;" : "=r"(output) : "r"(value), "r"(offset),"r"(mask));

    // CHECK: output = dpct::select_from_sub_group(item_ct1.get_sub_group(), value, offset);
    asm volatile("shfl.idx.b32 %0, %1, %2, 0xFFFFFFFF;" : "=r"(output) : "r"(value), "r"(offset));

   // CHECK: output = dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), value, offset);
    asm volatile("shfl.bfly.b32 %0, %1, %2, %3;" : "=r"(output) : "r"(value), "r"(offset),"r"(mask));

    // CHECK: output = dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), value, offset);
    asm volatile("shfl.bfly.b32 %0, %1, %2, 0xFFFFFFFF;" : "=r"(output) : "r"(value), "r"(offset));
}

// clang-format off
