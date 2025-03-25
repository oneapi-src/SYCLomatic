// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/mma %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/mma/mma.dp.cpp
// RUN: %if build_lit %{icpx -c -DNO_BUILD_TEST -fsycl %T/mma/mma.dp.cpp -o %T/mma/mma.dp.o %}

// clang-format off
#include <cuda_runtime.h>
#include <cuda_fp16.h>

/*
mma.sync.aligned.m16n8k16.alayout.blayout.dtype.f16.f16.ctype d, a, b, c;

Below are the currenly supported configurations:

.alayout = {.row};
.blayout = {.col};
.ctype   = {.f32};
.dtype   = {.f32};
*/

__global__ void mma_kernel() {
  int a[4];
  int b[2];
  float c[4];

  // CHECK: dpct::experimental::matrix::mma<sycl::half>(&c[0], &c[1], &c[2], &c[3], a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1], c[2], c[3], item_ct1);
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        " { %0, %1, %2, %3 }, "
        " { %4, %5, %6, %7 }, "
        " { %8, %9 }, "
        " { %0, %1, %2, %3 };"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
}


int main () {
  // CHECK: [=](sycl::nd_item<3> item_ct1) {{\[\[}}sycl::reqd_sub_group_size(32){{\]\]}} {
  mma_kernel<<<1, 32>>>();

  return 0;
}
// clang-format on
