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

---------     ---------   ----------   -----------   -------------
| Shape |     |   A   |   |    B   |   |  C / D  |   | Supported |
---------     ---------   ----------   -----------   -------------
m8n8k4          .f16         .f16       .f16/.f32        Yes
                .f64         .f64         .f64           Yes
m8n8k16        .s8/.u8      .s8/.u8       .s32           Yes
m8n8k32        .s4/.u4      .s4/.u4       .s32           Yes
m8n8k128        .b1          .b1          .s32           Yes

m16n8k4         .tf32        .tf32        .tf32          No
                .f64         .f64         .f64           Yes
m16n8k8      .f16/.bf16   .f16/.bf16    .f16/.f32        Partial (.f16.f16.f16.f16 / .f32.f16.f16.f32)
                .tf32        .tf32        .tf32          No
                .f64         .f64         .f64           Yes
m16n8k16     .f16/.bf16   .f16/.bf16    .f16/.f32        Partial (.f16.f16.f16.f16 / .f32.f16.f16.f32)
                .f64         .f64         .f64           Yes
              .s8/.u8      .s8/.u8        .s32           Yes
m16n8k32      .s4/.u4      .s4/.u4        .s32           Yes
              .s8/.u8      .s8/.u8        .s32           Yes
m16n8k64      .s4/.u4      .s4/.u4        .s32           Yes
m16n8k128       .b1          .b1          .s32           Yes
m16n8k256       .b1          .b1          .s32           Yes

A Layout: row
B Layout: col
*/

__global__ void mma_kernel_m8n8k4(int *a, int *b, float *c, double *d) {
  // CHECK: dpct::experimental::matrix::mma<8, 8, 4, sycl::half>(&c[0], &c[1], &c[2], &c[3], a[0], a[1], b[0], b[1], c[0], c[1], c[2], c[3]);
  asm("mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16 "
        " { %0, %1, %2, %3 }, "
        " { %4, %5 }, "
        " { %6, %7 }, "
        " { %8, %9, %10, %11 };"
        : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(b[1]),
          "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));

  // CHECK: dpct::experimental::matrix::mma<8, 8, 4, sycl::half>(&c[0], &c[1], &c[2], &c[3], &c[4], &c[5], &c[6], &c[7], a[0], a[1], b[0], b[1], c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]);
  asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        " { %0, %1, %2, %3, %4, %5, %6, %7 }, "
        " { %8, %9 }, "
        " { %10, %11 }, "
        " { %0, %1, %2, %3, %4, %5, %6, %7 };"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
        : "r"(a[0]), "r"(a[1]),
          "r"(b[0]), "r"(b[1]));

  // CHECK: dpct::experimental::matrix::mma<8, 8, 4, double>(&d[0], &d[1], a[0], b[0], d[0], d[1]);
  asm("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
        " { %0, %1 }, "
        " { %2 }, "
        " { %3 }, "
        " { %4, %5 };"
        : "=d"(d[0]), "=d"(d[1])
        : "d"(a[0]),
          "d"(b[0]),
          "d"(d[0]), "d"(d[1]));
}

__global__ void mma_kernel_m8n8k16(int *a, int *b, int *c, int *d) {
  // CHECK: dpct::experimental::matrix::mma<8, 8, 16, int8_t>(&d[0], &d[1], a[0], b[0], c[0], c[1]);
  asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
      " { %0, %1 }, "
      " { %2 }, "
      " { %3 }, "
      " { %4, %5 };"
      : "=r"(d[0]), "=r"(d[1])
      : "r"(a[0]),
        "r"(b[0]),
        "r"(c[0]), "r"(c[1]));

  // CHECK: dpct::experimental::matrix::mma<8, 8, 16, uint8_t>(&d[0], &d[1], a[0], b[0], c[0], c[1]);
  asm("mma.sync.aligned.m8n8k16.row.col.s32.u8.u8.s32 "
      " { %0, %1 }, "
      " { %2 }, "
      " { %3 }, "
      " { %4, %5 };"
      : "=r"(d[0]), "=r"(d[1])
      : "r"(a[0]),
        "r"(b[0]),
        "r"(c[0]), "r"(c[1]));
}

__global__ void mma_kernel_m8n8k32(int *a, int *b, int *c, int *d) {
  // CHECK: dpct::experimental::matrix::mma<8, 8, 32, int8_t>(&d[0], &d[1], a[0], b[0], c[0], c[1]);
  asm("mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 "
      " { %0, %1 }, "
      " { %2 }, "
      " { %3 }, "
      " { %4, %5 };"
      : "=r"(d[0]), "=r"(d[1])
      : "r"(a[0]),
        "r"(b[0]),
        "r"(c[0]), "r"(c[1]));

  // CHECK: dpct::experimental::matrix::mma<8, 8, 32, uint8_t>(&d[0], &d[1], a[0], b[0], c[0], c[1]);
  asm("mma.sync.aligned.m8n8k32.row.col.s32.u4.u4.s32 "
      " { %0, %1 }, "
      " { %2 }, "
      " { %3 }, "
      " { %4, %5 };"
      : "=r"(d[0]), "=r"(d[1])
      : "r"(a[0]),
        "r"(b[0]),
        "r"(c[0]), "r"(c[1]));
}

__global__ void mma_kernel_m8n8k128(int *a, int *b, int *c, int *d) {
  // CHECK: dpct::experimental::matrix::mma<8, 8, 128, uint8_t, sycl::bit_and<>>(&d[0], &d[1], a[0], b[0], c[0], c[1]);
  asm ("mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc "
      " { %0, %1 }, "
      " { %2 }, "
      " { %3 }, "
      " { %4, %5 };"
      : "=r"(d[0]), "=r"(d[1])
      : "r"(a[0]),
        "r"(b[0]),
        "r"(c[0]), "r"(c[1]));

  // CHECK: dpct::experimental::matrix::mma<8, 8, 128, uint8_t, sycl::bit_xor<>>(&d[0], &d[1], a[0], b[0], c[0], c[1]);
  asm ("mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.xor.popc "
      " { %0, %1 }, "
      " { %2 }, "
      " { %3 }, "
      " { %4, %5 };"
      : "=r"(d[0]), "=r"(d[1])
      : "r"(a[0]),
        "r"(b[0]),
        "r"(c[0]), "r"(c[1]));
}

__global__ void mma_kernel_m16n8k4(float *fa, float *fb, float *fc, float *fd, double *da, double *db, double *dc, double *dd) {
  // CHECK: dpct::experimental::matrix::mma<16, 8, 4, double>(&dd[0], &dd[1], &dd[2], &dd[3], da[0], da[1], db[0], dc[0], dc[1], dc[2], dc[3]);
  asm("mma.sync.aligned.m16n8k4.row.col.f64.f64.f64.f64 "
      " { %0, %1, %2, %3 }, "
      " { %4, %5 }, "
      " { %6 }, "
      " { %7, %8, %9, %10 };"
      : "=d"(dd[0]), "=d"(dd[1]), "=d"(dd[2]), "=d"(dd[3])
      : "d"(da[0]), "d"(da[1]),
        "d"(db[0]),
        "d"(dc[0]), "d"(dc[1]), "d"(dc[2]), "d"(dc[3]));
}

__global__ void mma_kernel_m16n8k8(int *a, int *b, uint *c, uint *d, float *fa, float *fb, float *fc, float *fd, double *da, double *db, double *dc, double *dd) {
  // CHECK: dpct::experimental::matrix::mma<16, 8, 8, sycl::half>(&d[0], &d[1], (*(reinterpret_cast<int *>(&a[0]))), (*(reinterpret_cast<int *>(&a[1]))), (*(reinterpret_cast<int *>(&b[0]))), c[0], c[1]);
  asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
      " { %0, %1 }, "
      " { %2, %3 }, "
      " { %4 }, "
      " { %5, %6 };"
      : "=r"(d[0]), "=r"(d[1])
      : "r"(*(reinterpret_cast<int *>(&a[0]))),
        "r"(*(reinterpret_cast<int *>(&a[1]))),
        "r"(*(reinterpret_cast<int *>(&b[0]))),
        "r"(c[0]), "r"(c[1]));

  // CHECK: dpct::experimental::matrix::mma<16, 8, 8, sycl::half>(&fd[0], &fd[1], &fd[2], &fd[3], *(reinterpret_cast<int *>(&a[0])), *(reinterpret_cast<int *>(&a[1])), *(reinterpret_cast<int *>(&b[0])), fc[0], fc[1], fc[2], fc[3]);
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

  // CHECK: dpct::experimental::matrix::mma<16, 8, 8, double>(&dd[0], &dd[1], &dd[2], &dd[3], da[0], da[1], da[2], da[3], db[0], db[1], dc[0], dc[1], dc[2], dc[3]);
  asm("mma.sync.aligned.m16n8k8.row.col.f64.f64.f64.f64 "
      " { %0, %1, %2, %3 }, "
      " { %4, %5, %6, %7 }, "
      " { %8, %9 }, "
      " { %10, %11, %12, %13 };"
      : "=d"(dd[0]), "=d"(dd[1]), "=d"(dd[2]), "=d"(dd[3])
      : "d"(da[0]), "d"(da[1]), "d"(da[2]), "d"(da[3]),
        "d"(db[0]), "d"(db[1]),
        "d"(dc[0]), "d"(dc[1]), "d"(dc[2]), "d"(dc[3]));
}

__global__ void mma_kernel_m16n8k16(uint *ua, uint *ub, uint *uc, uint *ud, int *a, int *b, int *c, float *fc, int *d, double *da, double *db, double *dc, double *dd) {
  // CHECK: dpct::experimental::matrix::mma<16, 8, 16, sycl::half>(&ud[0], &ud[1], a[0], a[1], a[2], a[3], b[0], b[1], uc[0], uc[1]);
  asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
      " { %0, %1 }, "
      " { %2, %3, %4, %5 }, "
      " { %6, %7 }, "
      " { %8, %9 };"
      : "=r"(ud[0]), "=r"(ud[1])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
        "r"(b[0]), "r"(b[1]),
        "r"(uc[0]), "r"(uc[1]));

  // CHECK: dpct::experimental::matrix::mma<16, 8, 16, sycl::half>(&fc[0], &fc[1], &fc[2], &fc[3], a[0], a[1], a[2], a[3], b[0], b[1], fc[0], fc[1], fc[2], fc[3]);
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        " { %0, %1, %2, %3 }, "
        " { %4, %5, %6, %7 }, "
        " { %8, %9 }, "
        " { %0, %1, %2, %3 };"
        : "+f"(fc[0]), "+f"(fc[1]), "+f"(fc[2]), "+f"(fc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));

  // CHECK: dpct::experimental::matrix::mma<16, 8, 16, double>(&dd[0], &dd[1], &dd[2], &dd[3], da[0], da[1], da[2], da[3], da[4], da[5], da[6], da[7], db[0], db[1], db[2], db[3], dc[0], dc[1], dc[2], dc[3]);  
  asm("mma.sync.aligned.m16n8k16.row.col.f64.f64.f64.f64 "
      " { %0, %1, %2, %3 }, "
      " { %4, %5, %6, %7, %8, %9, %10, %11 }, "
      " { %12, %13, %14, %15 }, "
      " { %16, %17, %18, %19 };"
      : "=d"(dd[0]), "=d"(dd[1]), "=d"(dd[2]), "=d"(dd[3])
      : "d"(da[0]), "d"(da[1]), "d"(da[2]), "d"(da[3]), "d"(da[4]), "d"(da[5]), "d"(da[6]), "d"(da[7]),
        "d"(db[0]), "d"(db[1]), "d"(db[2]), "d"(db[3]),
        "d"(dc[0]), "d"(dc[1]), "d"(dc[2]), "d"(dc[3]));

  // CHECK: dpct::experimental::matrix::mma<16, 8, 16, uint8_t>(&ud[0], &ud[1], &ud[2], &ud[3], ua[0], ua[1], ub[0], uc[0], uc[1], uc[2], uc[3]);
  asm("mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32 "
      " { %0, %1, %2, %3 }, "
      " { %4, %5 }, "
      " { %6 }, "
      " { %7, %8, %9, %10 };"
      : "=r"(ud[0]), "=r"(ud[1]), "=r"(ud[2]), "=r"(ud[3])
      : "r"(ua[0]), "r"(ua[1]),
        "r"(ub[0]),
        "r"(uc[0]), "r"(uc[1]), "r"(uc[2]), "r"(uc[3]));

  // CHECK: dpct::experimental::matrix::mma<16, 8, 16, int8_t>(&d[0], &d[1], &d[2], &d[3], a[0], a[1], b[0], c[0], c[1], c[2], c[3]);
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

__global__ void mma_kernel_m16n8k32(uint *ua, uint *ub, uint *uc, uint *ud, int *a, int *b, int *c, int *d) {
  // CHECK: dpct::experimental::matrix::mma<16, 8, 32, uint8_t>(&ud[0], &ud[1], &ud[2], &ud[3], ua[0], ua[1], ub[0], uc[0], uc[1], uc[2], uc[3]);
  asm("mma.sync.aligned.m16n8k32.row.col.s32.u4.u4.s32 "
      " { %0, %1, %2, %3 }, "
      " { %4, %5 }, "
      " { %6 }, "
      " { %7, %8, %9, %10 };"
      : "=r"(ud[0]), "=r"(ud[1]), "=r"(ud[2]), "=r"(ud[3])
      : "r"(ua[0]), "r"(ua[1]),
        "r"(ub[0]),
        "r"(uc[0]), "r"(uc[1]), "r"(uc[2]), "r"(uc[3]));

  // CHECK: dpct::experimental::matrix::mma<16, 8, 32, int8_t>(&d[0], &d[1], &d[2], &d[3], a[0], a[1], b[0], c[0], c[1], c[2], c[3]);
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32 "
      " { %0, %1, %2, %3 }, "
      " { %4, %5 }, "
      " { %6 }, "
      " { %7, %8, %9, %10 };"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a[0]), "r"(a[1]),
        "r"(b[0]),
        "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));

  // CHECK: dpct::experimental::matrix::mma<16, 8, 32, uint8_t>(&ud[0], &ud[1], &ud[2], &ud[3], ua[0], ua[1], ua[2], ua[3], ub[0], ub[1], uc[0], uc[1], uc[2], uc[3]);
  asm("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "
      " { %0, %1, %2, %3 }, "
      " { %4, %5, %6, %7 }, "
      " { %8, %9 }, "
      " { %10, %11, %12, %13 };"
      : "=r"(ud[0]), "=r"(ud[1]), "=r"(ud[2]), "=r"(ud[3])
      : "r"(ua[0]), "r"(ua[1]), "r"(ua[2]), "r"(ua[3]),
        "r"(ub[0]), "r"(ub[1]),
        "r"(uc[0]), "r"(uc[1]), "r"(uc[2]), "r"(uc[3]));

  // CHECK: dpct::experimental::matrix::mma<16, 8, 32, int8_t>(&d[0], &d[1], &d[2], &d[3], a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1], c[2], c[3]);
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

__global__ void mma_kernel_m16n8k64(uint *ua, uint *ub, uint *uc, uint *ud, int *a, int *b, int *c, int *d) {
  // CHECK: dpct::experimental::matrix::mma<16, 8, 64, uint8_t>(&ud[0], &ud[1], &ud[2], &ud[3], ua[0], ua[1], ua[2], ua[3], ub[0], ub[1], uc[0], uc[1], uc[2], uc[3]);
  asm("mma.sync.aligned.m16n8k64.row.col.s32.u4.u4.s32 "
      " { %0, %1, %2, %3 }, "
      " { %4, %5, %6, %7 }, "
      " { %8, %9 }, "
      " { %10, %11, %12, %13 };"
      : "=r"(ud[0]), "=r"(ud[1]), "=r"(ud[2]), "=r"(ud[3])
      : "r"(ua[0]), "r"(ua[1]), "r"(ua[2]), "r"(ua[3]),
        "r"(ub[0]), "r"(ub[1]),
        "r"(uc[0]), "r"(uc[1]), "r"(uc[2]), "r"(uc[3]));

  // CHECK: dpct::experimental::matrix::mma<16, 8, 64, int8_t>(&d[0], &d[1], &d[2], &d[3], a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1], c[2], c[3]);
  asm("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
      " { %0, %1, %2, %3 }, "
      " { %4, %5, %6, %7 }, "
      " { %8, %9 }, "
      " { %10, %11, %12, %13 };"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
        "r"(b[0]), "r"(b[1]),
        "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}

__global__ void mma_kernel_m16n8k128(int *a, int *b, int *c, int *d) {
  // CHECK: dpct::experimental::matrix::mma<16, 8, 128, uint8_t, sycl::bit_and<>>(&d[0], &d[1], &d[2], &d[3], a[0], a[1], b[0], c[0], c[1], c[2], c[3]);
  asm ("mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.and.popc "
      " { %0, %1, %2, %3 }, "
      " { %4, %5 }, "
      " { %6 }, "
      " { %7, %8, %9, %10 };"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a[0]), "r"(a[1]),
        "r"(b[0]),
        "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));

  // CHECK: dpct::experimental::matrix::mma<16, 8, 128, uint8_t, sycl::bit_xor<>>(&d[0], &d[1], &d[2], &d[3], a[0], a[1], b[0], c[0], c[1], c[2], c[3]);
  asm ("mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.xor.popc "
      " { %0, %1, %2, %3 }, "
      " { %4, %5 }, "
      " { %6 }, "
      " { %7, %8, %9, %10 };"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a[0]), "r"(a[1]),
        "r"(b[0]),
        "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}

__global__ void mma_kernel_m16n8k256(int *a, int *b, int *c, int *d) {
  // CHECK: dpct::experimental::matrix::mma<16, 8, 256, uint8_t, sycl::bit_and<>>(&d[0], &d[1], &d[2], &d[3], a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1], c[2], c[3]);
  asm ("mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
      " { %0, %1, %2, %3 }, "
      " { %4, %5, %6, %7 }, "
      " { %8, %9 }, "
      " { %10, %11, %12, %13 };"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
        "r"(b[0]), "r"(b[1]),
        "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));

  // CHECK: dpct::experimental::matrix::mma<16, 8, 256, uint8_t, sycl::bit_xor<>>(&d[0], &d[1], &d[2], &d[3], a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1], c[2], c[3]);
  asm ("mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc "
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
  uint *uint_a, *uint_b, *uint_c, *uint_d;
  int *int_a, *int_b, *int_c, *int_d;
  float *float_a, *float_b, *float_c, *float_d;
  double *double_a, *double_b, *double_c, *double_d;

  // CHECK: [=](sycl::nd_item<3> item_ct1) {{\[\[}}sycl::reqd_sub_group_size(32){{\]\]}} {
  mma_kernel_m8n8k4<<<1, 32>>>(int_a, int_b, float_c, double_d);
  // CHECK: [=](sycl::nd_item<3> item_ct1) {{\[\[}}sycl::reqd_sub_group_size(32){{\]\]}} {
  mma_kernel_m8n8k16<<<1, 32>>>(int_a, int_b, int_c, int_d);
  // CHECK: [=](sycl::nd_item<3> item_ct1) {{\[\[}}sycl::reqd_sub_group_size(32){{\]\]}} {
  mma_kernel_m8n8k32<<<1, 32>>>(int_a, int_b, int_c, int_d);
  // CHECK: [=](sycl::nd_item<3> item_ct1) {{\[\[}}sycl::reqd_sub_group_size(32){{\]\]}} {
  mma_kernel_m8n8k128<<<1, 32>>>(int_a, int_b, int_c, int_d);
  // CHECK: [=](sycl::nd_item<3> item_ct1) {{\[\[}}sycl::reqd_sub_group_size(32){{\]\]}} {
  mma_kernel_m16n8k4<<<1, 32>>>(float_a, float_b, float_c, float_d, double_a, double_b, double_c, double_d);
  // CHECK: [=](sycl::nd_item<3> item_ct1) {{\[\[}}sycl::reqd_sub_group_size(32){{\]\]}} {
  mma_kernel_m16n8k8<<<1, 32>>>(int_a, int_b, uint_c, uint_d, float_a, float_b, float_c, float_d, double_a, double_b, double_c, double_d);
  // CHECK: [=](sycl::nd_item<3> item_ct1) {{\[\[}}sycl::reqd_sub_group_size(32){{\]\]}} {
  mma_kernel_m16n8k16<<<1, 32>>>(uint_a, uint_b, uint_c, uint_d, int_a, int_b, int_c, float_c, int_d, double_a, double_b, double_c, double_d);
  // CHECK: [=](sycl::nd_item<3> item_ct1) {{\[\[}}sycl::reqd_sub_group_size(32){{\]\]}} {
  mma_kernel_m16n8k32<<<1, 32>>>(uint_a, uint_b, uint_c, uint_d, int_a, int_b, int_c, int_d);
  // CHECK: [=](sycl::nd_item<3> item_ct1) {{\[\[}}sycl::reqd_sub_group_size(32){{\]\]}} {
  mma_kernel_m16n8k64<<<1, 32>>>(uint_a, uint_b, uint_c, uint_d, int_a, int_b, int_c, int_d);
  // CHECK: [=](sycl::nd_item<3> item_ct1) {{\[\[}}sycl::reqd_sub_group_size(32){{\]\]}} {
  mma_kernel_m16n8k128<<<1, 32>>>(int_a, int_b, int_c, int_d);
  // CHECK: [=](sycl::nd_item<3> item_ct1) {{\[\[}}sycl::reqd_sub_group_size(32){{\]\]}} {
  mma_kernel_m16n8k256<<<1, 32>>>(int_a, int_b, int_c, int_d);

  return 0;
}
// clang-format on
