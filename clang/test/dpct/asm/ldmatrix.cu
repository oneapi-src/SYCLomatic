// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/ldmatrix %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/ldmatrix/ldmatrix.dp.cpp
// RUN: %if build_lit %{icpx -c -DNO_BUILD_TEST -fsycl %T/ldmatrix/ldmatrix.dp.cpp -o %T/ldmatrix/ldmatrix.dp.o %}

// clang-format off
#include <cuda_runtime.h>
#include <cuda_fp16.h>

/*
ldmatrix.sync.aligned.shape.num{.trans}{.ss}.type r, [p];

Below are the currenly supported configurations:
.shape = {.m8n8};
.num   = {.x1, .x2, .x4};
.ss    = {.shared{::cta}};
.type  = {.b16};
*/

__device__ void load_matrix_x1(void *sh_r_addr, int *r) {
  // CHECK: auto addr = sh_r_addr;
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(sh_r_addr));

  // CHECK: dpct::experimental::matrix::ldmatrix((uintptr_t)addr, &r[0]);
  asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                : "=r"(r[0])
                : "r"(addr));
}

__device__ void load_matrix_x2(void *sh_r_addr, int *r) {
  // CHECK: auto addr = sh_r_addr;
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(sh_r_addr));

  // CHECK: dpct::experimental::matrix::ldmatrix((uintptr_t)addr, &r[0], &r[1]);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                : "=r"(r[0]), "=r"(r[1])
                : "r"(addr));
}

__device__ void load_matrix_x4(void *sh_r_addr, int *r) {
  // CHECK: auto addr = sh_r_addr;
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(sh_r_addr));

  // CHECK: dpct::experimental::matrix::ldmatrix((uintptr_t)addr, &r[0], &r[1], &r[2], &r[3]);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                : "r"(addr));
}

__device__ void load_matrix_x1_trans(void *sh_r_addr, int *r) {
  // CHECK: auto addr = sh_r_addr;
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(sh_r_addr));

  // CHECK: dpct::experimental::matrix::ldmatrix((uintptr_t)addr, &r[0], true);
  asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
                : "=r"(r[0])
                : "r"(addr));
}

__device__ void load_matrix_x2_trans(void *sh_r_addr, int *r) {
  // CHECK: auto addr = sh_r_addr;
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(sh_r_addr));

  // CHECK: dpct::experimental::matrix::ldmatrix((uintptr_t)addr, &r[0], &r[1], true);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                : "=r"(r[0]), "=r"(r[1])
                : "r"(addr));
}

__device__ void load_matrix_x4_trans(void *sh_r_addr, int *r) {
  // CHECK: auto addr = sh_r_addr;
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(sh_r_addr));

  // CHECK: dpct::experimental::matrix::ldmatrix((uintptr_t)addr, &r[0], &r[1], &r[2], &r[3], true);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                : "r"(addr));
}

__global__ void load_kernel() {
  __shared__ half s_data[1024];
  int r[4];

  load_matrix_x1(s_data, r);
  load_matrix_x2(s_data, r);
  load_matrix_x4(s_data, r);
  load_matrix_x1_trans(s_data, r);
  load_matrix_x2_trans(s_data, r);
  load_matrix_x4_trans(s_data, r);
}

int main () {
  // CHECK: [=](sycl::nd_item<3> item_ct1) {{\[\[}}sycl::reqd_sub_group_size(32){{\]\]}} {
  load_kernel<<<1, 32>>>();

  return 0;
}

#ifndef NO_BUILD_TEST
__device__ void test_xn(uint32_t addr, int *r) {
  // CHECK: DPCT1053:{{.*}}: Migration of device assembly code is not supported.
  asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0, %1}, [%2];\n"
                : "=r"(r[0]), "=r"(r[1])
                : "r"(addr));

  // CHECK: DPCT1053:{{.*}}: Migration of device assembly code is not supported.
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0}, [%0];\n"
                :
                : "r"(addr));

  // CHECK: DPCT1053:{{.*}}: Migration of device assembly code is not supported.
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2}, [%3];\n"
                : "=r"(r[0]), "=r"(r[1]), "=r"(r[2])
                : "r"(addr));
}
#endif // NO_BUILD_TEST

// clang-format on
