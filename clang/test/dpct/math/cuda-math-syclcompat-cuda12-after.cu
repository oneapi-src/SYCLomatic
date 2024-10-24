// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-11.8
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6, v11.7, v11.8
// RUN: dpct --format-range=none -use-syclcompat -out-root %T/math/cuda-math-syclcompat-cuda12-after %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14
// RUN: FileCheck --input-file %T/math/cuda-math-syclcompat-cuda12-after/cuda-math-syclcompat-cuda12-after.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/math/cuda-math-syclcompat-cuda12-after/cuda-math-syclcompat-cuda12-after.dp.cpp -o %T/math/cuda-math-syclcompat-cuda12-after/cuda-math-syclcompat-cuda12-after.dp.o %}

#include "cuda_fp16.h"

using namespace std;

__global__ void kernelFuncSIMD() {
  unsigned int u, u_1, u_2, u_3;
  int i, i_1, i_2, i_3;
  bool b_1, b_2;
  // CHECK: u_3 = syclcompat::vectorized_ternary<sycl::short2>(u, u_1, u_2, std::plus<>(), syclcompat::maximum());
  u_3 = __viaddmax_s16x2(u, u_1, u_2);
  // CHECK: u_3 = syclcompat::vectorized_ternary<sycl::short2>(u, u_1, u_2, std::plus<>(), syclcompat::maximum(), true);
  u_3 = __viaddmax_s16x2_relu(u, u_1, u_2);
  // CHECK: i_3 = sycl::max<int>(i + i_1, i_2);
  i_3 = __viaddmax_s32(i, i_1, i_2);
  // CHECK: i_3 = syclcompat::relu<int>(sycl::max<int>(i + i_1, i_2));
  i_3 = __viaddmax_s32_relu(i, i_1, i_2);
  // CHECK: u_3 = syclcompat::vectorized_ternary<sycl::ushort2>(u, u_1, u_2, std::plus<>(), syclcompat::maximum());
  u_3 = __viaddmax_u16x2(u, u_1, u_2);
  // CHECK: u_3 = sycl::max<unsigned>(u + u_1, u_2);
  u_3 = __viaddmax_u32(u, u_1, u_2);
  // CHECK: u_3 = syclcompat::vectorized_ternary<sycl::short2>(u, u_1, u_2, std::plus<>(), syclcompat::minimum());
  u_3 = __viaddmin_s16x2(u, u_1, u_2);
  // CHECK: u_3 = syclcompat::vectorized_ternary<sycl::short2>(u, u_1, u_2, std::plus<>(), syclcompat::minimum(), true);
  u_3 = __viaddmin_s16x2_relu(u, u_1, u_2);
  // CHECK: i_3 = sycl::min<int>(i + i_1, i_2);
  i_3 = __viaddmin_s32(i, i_1, i_2);
  // CHECK: i_3 = syclcompat::relu<int>(sycl::min<int>(i + i_1, i_2));
  i_3 = __viaddmin_s32_relu(i, i_1, i_2);
  // CHECK: u_3 = syclcompat::vectorized_ternary<sycl::ushort2>(u, u_1, u_2, std::plus<>(), syclcompat::minimum());
  u_3 = __viaddmin_u16x2(u, u_1, u_2);
  // CHECK: u_3 = sycl::min<unsigned>(u + u_1, u_2);
  u_3 = __viaddmin_u32(u, u_1, u_2);
  // CHECK: u_2 = syclcompat::vectorized_with_pred<short>(u, u_1, syclcompat::maximum(), &b_1, &b_2);
  u_2 = __vibmax_s16x2(u, u_1, &b_1, &b_2);
  // CHECK: i_2 = syclcompat::maximum()(i, i_1, &b_1);
  i_2 = __vibmax_s32(i, i_1, &b_1);
  // CHECK: u_2 = syclcompat::vectorized_with_pred<unsigned short>(u, u_1, syclcompat::maximum(), &b_1, &b_2);
  u_2 = __vibmax_u16x2(u, u_1, &b_1, &b_2);
  // CHECK: u_2 = syclcompat::maximum()(u, u_1, &b_1);
  u_2 = __vibmax_u32(u, u_1, &b_1);
  // CHECK: u_2 = syclcompat::vectorized_with_pred<short>(u, u_1, syclcompat::minimum(), &b_1, &b_2);
  u_2 = __vibmin_s16x2(u, u_1, &b_1, &b_2);
  // CHECK: i_2 = syclcompat::minimum()(i, i_1, &b_1);
  i_2 = __vibmin_s32(i, i_1, &b_1);
  // CHECK: u_2 = syclcompat::vectorized_with_pred<unsigned short>(u, u_1, syclcompat::minimum(), &b_1, &b_2);
  u_2 = __vibmin_u16x2(u, u_1, &b_1, &b_2);
  // CHECK: u_2 = syclcompat::minimum()(u, u_1, &b_1);
  u_2 = __vibmin_u32(u, u_1, &b_1);
  // CHECK: u_3 = syclcompat::vectorized_ternary<sycl::short2>(u, u_1, u_2, syclcompat::maximum(), syclcompat::maximum());
  u_3 = __vimax3_s16x2(u, u_1, u_2);
  // CHECK: u_3 = syclcompat::vectorized_ternary<sycl::short2>(u, u_1, u_2, syclcompat::maximum(), syclcompat::maximum(), true);
  u_3 = __vimax3_s16x2_relu(u, u_1, u_2);
  // CHECK: i_3 = sycl::max<int>(sycl::max<int>(i, i_1), i_2);
  i_3 = __vimax3_s32(i, i_1, i_2);
  // CHECK: i_3 = syclcompat::relu<int>(sycl::max<int>(sycl::max<int>(i, i_1), i_2));
  i_3 = __vimax3_s32_relu(i, i_1, i_2);
  // CHECK: u_3 = syclcompat::vectorized_ternary<sycl::ushort2>(u, u_1, u_2, syclcompat::maximum(), syclcompat::maximum());
  u_3 = __vimax3_u16x2(u, u_1, u_2);
  // CHECK: u_3 = sycl::max<unsigned>(sycl::max<unsigned>(u, u_1), u_2);
  u_3 = __vimax3_u32(u, u_1, u_2);
  // CHECK: u_2 = syclcompat::vectorized_binary<sycl::short2>(u, u_1, syclcompat::maximum(), true);
  u_2 = __vimax_s16x2_relu(u, u_1);
  // CHECK: i_2 = syclcompat::relu<int>(sycl::max<int>(i, i_1));
  i_2 = __vimax_s32_relu(i, i_1);
  // CHECK: u_3 = syclcompat::vectorized_ternary<sycl::short2>(u, u_1, u_2, syclcompat::minimum(), syclcompat::minimum());
  u_3 = __vimin3_s16x2(u, u_1, u_2);
  // CHECK: u_3 = syclcompat::vectorized_ternary<sycl::short2>(u, u_1, u_2, syclcompat::minimum(), syclcompat::minimum(), true);
  u_3 = __vimin3_s16x2_relu(u, u_1, u_2);
  // CHECK: i_3 = sycl::min<int>(sycl::min<int>(i, i_1), i_2);
  i_3 = __vimin3_s32(i, i_1, i_2);
  // CHECK: i_3 = syclcompat::relu<int>(sycl::min<int>(sycl::min<int>(i, i_1), i_2));
  i_3 = __vimin3_s32_relu(i, i_1, i_2);
  // CHECK: u_3 = syclcompat::vectorized_ternary<sycl::ushort2>(u, u_1, u_2, syclcompat::minimum(), syclcompat::minimum());
  u_3 = __vimin3_u16x2(u, u_1, u_2);
  // CHECK: u_3 = sycl::min<unsigned>(sycl::min<unsigned>(u, u_1), u_2);
  u_3 = __vimin3_u32(u, u_1, u_2);
  // CHECK: u_2 = syclcompat::vectorized_binary<sycl::short2>(u, u_1, syclcompat::minimum(), true);
  u_2 = __vimin_s16x2_relu(u, u_1);
  // CHECK: i_2 = syclcompat::relu<int>(sycl::min<int>(i, i_1));
  i_2 = __vimin_s32_relu(i, i_1);
}

int main() { return 0; }
