// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/mma %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/mma/mma.dp.cpp
// RUN: %if build_lit %{icpx -c -DNO_BUILD_TEST -fsycl %T/mma/mma.dp.cpp -o %T/mma/mma.dp.o %}

// clang-format off
#include <cuda_runtime.h>
#include <cuda_fp16.h>

/*
As per PTX ASM 8.1, below is the status of supported configurations

---------     ---------   ----------   -----------
| Shape |     |   A   |   |    B   |   |  C / D  |
---------     ---------   ----------   -----------
m8n8k4          .f16         .f16         .f32    
m8n8k16         .s8          .s8          .s32
m16n8k8       .f16/.bf16  .f16/.bf16      .f32    
m16n8k16        .f16         .f16         .f32
                .bf16        .bf16        .f32
                .s8          .s8          .s32  
                .f16         .f16         .f16     
m16n8k32        .s8          .s8          .s32    

Except for m8n8k4, all other shapes are supported for row/col layout of A/B matrices respectively.
*/

__global__ void mma_kernel_m8n8k4(int *a, int *b, float *c) {
  // CHECK: {
  // CHECK-NEXT:   volatile void *d_mat_frag_ct1[8] = { &c[0], &c[1], &c[2], &c[3], &c[4], &c[5], &c[6], &c[7] };
  // CHECK-NEXT:   sycl::vec<uint32_t, 2> a_mat_frag_ct1(a[0], a[1]);
  // CHECK-NEXT:   sycl::vec<uint32_t, 2> b_mat_frag_ct1(b[0], b[1]);
  // CHECK-NEXT:   sycl::vec<float, 8> c_mat_frag_ct1(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]);
  // CHECK-NEXT:   dpct::experimental::matrix::mma<8, 8, 4, sycl::half, float>(reinterpret_cast<volatile void **>(d_mat_frag_ct1), &a_mat_frag_ct1, &b_mat_frag_ct1, &c_mat_frag_ct1);
  // CHECK-NEXT: }
  asm("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 "
        " { %0, %1, %2, %3, %4, %5, %6, %7 }, "
        " { %8, %9 }, "
        " { %10, %11 }, "
        " { %0, %1, %2, %3, %4, %5, %6, %7 };"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
        : "r"(a[0]), "r"(a[1]),
          "r"(b[0]), "r"(b[1]));
}

__global__ void mma_kernel_m8n8k16(int *a, int *b, int *c, int *d) {
  // CHECK: {
  // CHECK-NEXT:   volatile void *d_mat_frag_ct1[2] = { &d[0], &d[1] };
  // CHECK-NEXT:   sycl::vec<uint32_t, 1> a_mat_frag_ct1(a[0]);
  // CHECK-NEXT:   sycl::vec<uint32_t, 1> b_mat_frag_ct1(b[0]);
  // CHECK-NEXT:   sycl::vec<int32_t, 2> c_mat_frag_ct1(c[0], c[1]);
  // CHECK-NEXT:   dpct::experimental::matrix::mma<8, 8, 16, int8_t, int32_t>(reinterpret_cast<volatile void **>(d_mat_frag_ct1), &a_mat_frag_ct1, &b_mat_frag_ct1, &c_mat_frag_ct1);
  // CHECK-NEXT: }
  asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
      " { %0, %1 }, "
      " { %2 }, "
      " { %3 }, "
      " { %4, %5 };"
      : "=r"(d[0]), "=r"(d[1])
      : "r"(a[0]),
        "r"(b[0]),
        "r"(c[0]), "r"(c[1]));
}

__global__ void mma_kernel_m16n8k8(int *a, int *b, float *fc, float *fd) {
  // CHECK: {
  // CHECK-NEXT:   volatile void *d_mat_frag_ct1[4] = { &fd[0], &fd[1], &fd[2], &fd[3] };
  // CHECK-NEXT:   sycl::vec<uint32_t, 2> a_mat_frag_ct1(*(reinterpret_cast<int *>(&a[0])), *(reinterpret_cast<int *>(&a[1])));
  // CHECK-NEXT:   sycl::vec<uint32_t, 1> b_mat_frag_ct1(*(reinterpret_cast<int *>(&b[0])));
  // CHECK-NEXT:   sycl::vec<float, 4> c_mat_frag_ct1(fc[0], fc[1], fc[2], fc[3]);
  // CHECK-NEXT:   dpct::experimental::matrix::mma<16, 8, 8, sycl::ext::oneapi::bfloat16, float>(reinterpret_cast<volatile void **>(d_mat_frag_ct1), &a_mat_frag_ct1, &b_mat_frag_ct1, &c_mat_frag_ct1);
  // CHECK-NEXT: }
  asm("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
      " { %0, %1, %2, %3 }, "
      " { %4, %5 }, "
      " { %6 }, "
      " { %7, %8, %9, %10 };"
      : "=f"(fd[0]), "=f"(fd[1]), "=f"(fd[2]), "=f"(fd[3])
      : "r"(*(reinterpret_cast<int *>(&a[0]))),
        "r"(*(reinterpret_cast<int *>(&a[1]))),
        "r"(*(reinterpret_cast<int *>(&b[0]))),
        "f"(fc[0]), "f"(fc[1]), "f"(fc[2]), "f"(fc[3]));

  // CHECK: {
  // CHECK-NEXT:   volatile void *d_mat_frag_ct1[4] = { &fd[0], &fd[1], &fd[2], &fd[3] };
  // CHECK-NEXT:   sycl::vec<uint32_t, 2> a_mat_frag_ct1(*(reinterpret_cast<int *>(&a[0])), *(reinterpret_cast<int *>(&a[1])));
  // CHECK-NEXT:   sycl::vec<uint32_t, 1> b_mat_frag_ct1(*(reinterpret_cast<int *>(&b[0])));
  // CHECK-NEXT:   sycl::vec<float, 4> c_mat_frag_ct1(fc[0], fc[1], fc[2], fc[3]);
  // CHECK-NEXT:   dpct::experimental::matrix::mma<16, 8, 8, sycl::half, float>(reinterpret_cast<volatile void **>(d_mat_frag_ct1), &a_mat_frag_ct1, &b_mat_frag_ct1, &c_mat_frag_ct1);
  // CHECK-NEXT: }
  asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
      " { %0, %1, %2, %3 }, "
      " { %4, %5 }, "
      " { %6 }, "
      " { %7, %8, %9, %10 };"
      : "=f"(fd[0]), "=f"(fd[1]), "=f"(fd[2]), "=f"(fd[3])
      : "r"(*(reinterpret_cast<int *>(&a[0]))),
        "r"(*(reinterpret_cast<int *>(&a[1]))),
        "r"(*(reinterpret_cast<int *>(&b[0]))),
        "f"(fc[0]), "f"(fc[1]), "f"(fc[2]), "f"(fc[3]));
}

__global__ void mma_kernel_m16n8k16(int *a, int *b, int *c, int *d) {
  // CHECK: {
  // CHECK-NEXT:   volatile void *d_mat_frag_ct1[2] = { &d[0], &d[1] };
  // CHECK-NEXT:   sycl::vec<uint32_t, 4> a_mat_frag_ct1(a[0], a[1], a[2], a[3]);
  // CHECK-NEXT:   sycl::vec<uint32_t, 2> b_mat_frag_ct1(b[0], b[1]);
  // CHECK-NEXT:   sycl::vec<uint32_t, 2> c_mat_frag_ct1(c[0], c[1]);
  // CHECK-NEXT:   dpct::experimental::matrix::mma<16, 8, 16, sycl::half, sycl::half>(reinterpret_cast<volatile void **>(d_mat_frag_ct1), &a_mat_frag_ct1, &b_mat_frag_ct1, &c_mat_frag_ct1);
  // CHECK-NEXT: }
  asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        " { %0, %1 }, "
        " { %2, %3, %4, %5 }, "
        " { %6, %7 }, "
        " { %8, %9 };"
        : "+r"(d[0]), "+r"(d[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "r"(c[0]), "r"(c[1]));
}

__global__ void mma_kernel_m16n8k16(int *a, int *b, int *c, float *fc, int *d) {
  // CHECK: {
  // CHECK-NEXT:   volatile void *d_mat_frag_ct1[4] = { &fc[0], &fc[1], &fc[2], &fc[3] };
  // CHECK-NEXT:   sycl::vec<uint32_t, 4> a_mat_frag_ct1(a[0], a[1], a[2], a[3]);
  // CHECK-NEXT:   sycl::vec<uint32_t, 2> b_mat_frag_ct1(b[0], b[1]);
  // CHECK-NEXT:   sycl::vec<float, 4> c_mat_frag_ct1(fc[0], fc[1], fc[2], fc[3]);
  // CHECK-NEXT:   dpct::experimental::matrix::mma<16, 8, 16, sycl::half, float>(reinterpret_cast<volatile void **>(d_mat_frag_ct1), &a_mat_frag_ct1, &b_mat_frag_ct1, &c_mat_frag_ct1);
  // CHECK-NEXT: }
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        " { %0, %1, %2, %3 }, "
        " { %4, %5, %6, %7 }, "
        " { %8, %9 }, "
        " { %0, %1, %2, %3 };"
        : "+f"(fc[0]), "+f"(fc[1]), "+f"(fc[2]), "+f"(fc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));

  // CHECK: {
  // CHECK-NEXT:   volatile void *d_mat_frag_ct1[4] = { &fc[0], &fc[1], &fc[2], &fc[3] };
  // CHECK-NEXT:   sycl::vec<uint32_t, 4> a_mat_frag_ct1(a[0], a[1], a[2], a[3]);
  // CHECK-NEXT:   sycl::vec<uint32_t, 2> b_mat_frag_ct1(b[0], b[1]);
  // CHECK-NEXT:   sycl::vec<float, 4> c_mat_frag_ct1(fc[0], fc[1], fc[2], fc[3]);
  // CHECK-NEXT:   dpct::experimental::matrix::mma<16, 8, 16, sycl::ext::oneapi::bfloat16, float>(reinterpret_cast<volatile void **>(d_mat_frag_ct1), &a_mat_frag_ct1, &b_mat_frag_ct1, &c_mat_frag_ct1);
  // CHECK-NEXT: }
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        " { %0, %1, %2, %3 }, "
        " { %4, %5, %6, %7 }, "
        " { %8, %9 }, "
        " { %0, %1, %2, %3 };"
        : "+f"(fc[0]), "+f"(fc[1]), "+f"(fc[2]), "+f"(fc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));

  // CHECK: {
  // CHECK-NEXT:   volatile void *d_mat_frag_ct1[4] = { &d[0], &d[1], &d[2], &d[3] };
  // CHECK-NEXT:   sycl::vec<uint32_t, 2> a_mat_frag_ct1(a[0], a[1]);
  // CHECK-NEXT:   sycl::vec<uint32_t, 1> b_mat_frag_ct1(b[0]);
  // CHECK-NEXT:   sycl::vec<int32_t, 4> c_mat_frag_ct1(c[0], c[1], c[2], c[3]);
  // CHECK-NEXT:   dpct::experimental::matrix::mma<16, 8, 16, int8_t, int32_t>(reinterpret_cast<volatile void **>(d_mat_frag_ct1), &a_mat_frag_ct1, &b_mat_frag_ct1, &c_mat_frag_ct1);
  // CHECK-NEXT: }
  asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
      " { %0, %1, %2, %3 }, "
      " { %4, %5 }, "
      " { %6 }, "
      " { %7, %8, %9, %10 };"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a[0]), "r"(a[1]),
        "r"(b[0]),
        "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}

__global__ void mma_kernel_m16n8k32(int *a, int *b, int *c, int *d) {
  // CHECK: {
  // CHECK-NEXT:   volatile void *d_mat_frag_ct1[4] = { &d[0], &d[1], &d[2], &d[3] };
  // CHECK-NEXT:   sycl::vec<uint32_t, 4> a_mat_frag_ct1(a[0], a[1], a[2], a[3]);
  // CHECK-NEXT:   sycl::vec<uint32_t, 2> b_mat_frag_ct1(b[0], b[1]);
  // CHECK-NEXT:   sycl::vec<int32_t, 4> c_mat_frag_ct1(c[0], c[1], c[2], c[3]);
  // CHECK-NEXT:   dpct::experimental::matrix::mma<16, 8, 32, int8_t, int32_t>(reinterpret_cast<volatile void **>(d_mat_frag_ct1), &a_mat_frag_ct1, &b_mat_frag_ct1, &c_mat_frag_ct1);
  // CHECK-NEXT: }
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      " { %0, %1, %2, %3 }, "
      " { %4, %5, %6, %7 }, "
      " { %8, %9 }, "
      " { %10, %11, %12, %13 };"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
        "r"(b[0]), "r"(b[1]),
        "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}


int main () {
  int *int_a, *int_b, *int_c, *int_d;
  float *float_c, *float_d;

  // CHECK: [=](sycl::nd_item<3> item_ct1) {{\[\[}}sycl::reqd_sub_group_size(32){{\]\]}} {
  mma_kernel_m8n8k4<<<1, 32>>>(int_a, int_b, float_c);
  // CHECK: [=](sycl::nd_item<3> item_ct1) {{\[\[}}sycl::reqd_sub_group_size(32){{\]\]}} {
  mma_kernel_m8n8k16<<<1, 32>>>(int_a, int_b, int_c, int_d);
  // CHECK: [=](sycl::nd_item<3> item_ct1) {{\[\[}}sycl::reqd_sub_group_size(32){{\]\]}} {
  mma_kernel_m16n8k8<<<1, 32>>>(int_a, int_b, float_c, float_d);
  // CHECK: [=](sycl::nd_item<3> item_ct1) {{\[\[}}sycl::reqd_sub_group_size(32){{\]\]}} {
  mma_kernel_m16n8k16<<<1, 32>>>(int_a, int_b, int_c, float_c, int_d);
  // CHECK: [=](sycl::nd_item<3> item_ct1) {{\[\[}}sycl::reqd_sub_group_size(32){{\]\]}} {
  mma_kernel_m16n8k32<<<1, 32>>>(int_a, int_b, int_c, int_d);

  return 0;
}
// clang-format on
