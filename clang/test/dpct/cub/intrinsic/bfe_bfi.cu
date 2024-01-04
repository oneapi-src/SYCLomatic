// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct -in-root %S -out-root %T/intrinsic/bfe_bfi %S/bfe_bfi.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/intrinsic/bfe_bfi/bfe_bfi.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/intrinsic/bfe_bfi/bfe_bfi.dp.cpp -o %T/intrinsic/bfe_bfi/bfe_bfi.dp.o %}

// CHECK:#include <sycl/sycl.hpp>
// CHECK:#include <dpct/dpct.hpp>
#include <cub/cub.cuh>
#include <cstdint>

__global__ void kernel() {
  int32_t i32 = 0;
  uint32_t u32 = 0;
  int64_t i64 = 0;
  uint64_t u64 = 0;
  uint32_t bit_start = 1;
  uint32_t num_bits = 4;
  uint32_t res;

  // CHECK: dpct::bfe_safe(i32, bit_start, num_bits);
  // CHECK-NEXT: dpct::bfe_safe(u32, bit_start, num_bits);
  // CHECK-NEXT: dpct::bfe_safe(i64, bit_start, num_bits);
  // CHECK-NEXT: dpct::bfe_safe(u64, bit_start, num_bits);
  // CHECK-NEXT: res = dpct::bfi_safe<unsigned>(res, u32, u32, bit_start);
  cub::BFE(i32, bit_start, num_bits);
  cub::BFE(u32, bit_start, num_bits);
  cub::BFE(i64, bit_start, num_bits);
  cub::BFE(u64, bit_start, num_bits);
  cub::BFI(res, u32, u32, bit_start, num_bits);
}
