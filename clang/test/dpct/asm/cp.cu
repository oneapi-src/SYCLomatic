// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/cp %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/atom/cp.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/red/cp.dp.cpp -o %T/red/cp.dp.o %}

// clang-format off
#include <cuda_runtime.h>
#include <cstdint>

/*
Does not support this
.space =              { .global, .shared{::cta, ::cluster} };
.type =               { .b32, .b64, .u32, .u64, .s32, .s64, .f32, .f64 };
.scope =              { .cta, .cluster, .gpu, .sys };
.sem =                { .release, .relaxed };


*/

__global__ void cp(int *a) {
  int a = 0;
  
  // CHECK: dpct::get_current_device().in_order_queue().memcpy(smem, glob_ptr, BYTES);
  asm volatile ("cp.async.cg.shared.global  [%0], [%1], %2;" : :  "r"(smem), "l"(glob_ptr), "n"(BYTES));

  
  
}

// clang-format on
