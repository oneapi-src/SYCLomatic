// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/shfl_sync %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/shfl_sync/shfl_sync.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/shfl_sync/shfl_sync.dp.cpp -o %T/shfl_sync/shfl_sync.dp.o %}

// clang-format off
#include <cuda_runtime.h>
#include <cstdint>

__global__ void shfl_sync() {
    int value; 
    unsigned mask = 0xFFFFFFFF;
    int offset;
    int output;                                              

    // CHECK:    /*
    // CHECK-NEXT:    DPCT1023:{{[0-9]+}}: The SYCL sub-group does not support mask options for shift_sub_group_right. You can specify "--use-experimental-features=masked-sub-group-operation" to use the experimental helper function to migrate inline asm instruction "shfl.sync.up.b32 %0, %1, %2, %3, %4;".
    // CHECK-NEXT:    */
    // CHECK-NEXT:    output = dpct::shift_sub_group_right(item_ct1.get_sub_group(), value, offset);
    asm volatile("shfl.sync.up.b32 %0, %1, %2, %3, %4;" : "=r"(output) : "r"(value), "r"(offset), "r"(0), "r"(mask));

    // CHECK:    /*
    // CHECK-NEXT:    DPCT1023:{{[0-9]+}}: The SYCL sub-group does not support mask options for shift_sub_group_left. You can specify "--use-experimental-features=masked-sub-group-operation" to use the experimental helper function to migrate inline asm instruction "shfl.sync.down.b32 %0, %1, %2, %3, %4;".
    // CHECK-NEXT:    */
    // CHECK-NEXT    output = dpct::shift_sub_group_left(item_ct1.get_sub_group(), value, offset);  
    asm volatile("shfl.sync.down.b32 %0, %1, %2, %3, %4;" : "=r"(output) : "r"(value), "r"(offset), "r"(0), "r"(mask));

    // CHECK:    /*
    // CHECK-NEXT:    DPCT1023:{{[0-9]+}}: The SYCL sub-group does not support mask options for select_from_sub_group. You can specify "--use-experimental-features=masked-sub-group-operation" to use the experimental helper function to migrate inline asm instruction "shfl.sync.idx.b32 %0, %1, %2, %3, %4;".
    // CHECK-NEXT:    */ 
    // CHECK-NEXT    output = dpct::select_from_sub_group(item_ct1.get_sub_group(), value, offset);  
    asm volatile("shfl.sync.idx.b32 %0, %1, %2, %3, %4;" : "=r"(output) : "r"(value), "r"(offset), "r"(0), "r"(mask));

    // CHECK:    /*
    // CHECK-NEXT:    DPCT1023:{{[0-9]+}}: The SYCL sub-group does not support mask options for permute_sub_group_by_xor. You can specify "--use-experimental-features=masked-sub-group-operation" to use the experimental helper function to migrate inline asm instruction "shfl.sync.bfly.b32 %0, %1, %2, %3, %4;".
    // CHECK-NEXT:    */
    // CHECK-NEXT:    output = dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), value, offset); 
    asm volatile("shfl.sync.bfly.b32 %0, %1, %2, %3, %4;" : "=r"(output) : "r"(value), "r"(offset), "r"(0), "r"(mask));
}

// clang-format off
