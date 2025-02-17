// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -out-root %T/cvt %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/cvt/cvt.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/cvt/cvt.dp.cpp -o %T/cvt/cvt.dp.o %}
// RUN: rm -rf %T/cvt/
// RUN: dpct --format-range=none --use-dpcpp-extensions=intel_device_math -out-root %T/cvt %s --cuda-include-path="%cuda-path/include" --extra-arg="-DUSE_INTEL_DEVICE_MATH" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/cvt/cvt.dp.cpp --check-prefix=CHECK-INTEL-EXT
// RUN: %if build_lit %{icpx -c -DUSE_INTEL_DEVICE_MATH -fsycl %T/cvt/cvt.dp.cpp -o %T/cvt/cvt.dp.o %}

// clang-format off
// CHECK-INTEL-EXT: #include <sycl/ext/intel/math.hpp>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

#ifdef USE_INTEL_DEVICE_MATH
__global__ void cvt() {
  uint16_t u16;
  uint32_t u32;
  uint64_t u64;
  int16_t s16;
  int32_t s32;
  float f32;
  double f64;

  // Test half to integer conversion with rni
  // CHECK-INTEL-EXT: u64 = sycl::ext::intel::math::half2ull_rn(sycl::vec<uint16_t, 1>(u16).template as<sycl::vec<sycl::half, 1>>().x());
  // CHECK-INTEL-EXT: u32 = sycl::ext::intel::math::half2uint_rn(sycl::vec<uint16_t, 1>(u16).template as<sycl::vec<sycl::half, 1>>().x());
  // CHECK-INTEL-EXT: u16 = sycl::ext::intel::math::half2ushort_rn(sycl::vec<uint16_t, 1>(u16).template as<sycl::vec<sycl::half, 1>>().x());
  // CHECK-INTEL-EXT: u32 = sycl::ext::intel::math::half2ushort_rn(sycl::vec<uint16_t, 1>(u16).template as<sycl::vec<sycl::half, 1>>().x());
  asm volatile ("cvt.rni.u64.f16 %0, %1;" : "=l"(u64) : "h"(u16));
  asm volatile ("cvt.rni.u32.f16 %0, %1;" : "=r"(u32) : "h"(u16));
  asm volatile ("cvt.rni.u16.f16 %0, %1;" : "=h"(u16) : "h"(u16));
  asm volatile ("cvt.rni.u8.f16 %0, %1;" : "=r"(u32) : "h"(u16));

  // Test half to integer conversion with rzi
  // CHECK-INTEL-EXT: u64 = sycl::ext::intel::math::half2ull_rz(sycl::vec<uint16_t, 1>(u16).template as<sycl::vec<sycl::half, 1>>().x());
  // CHECK-INTEL-EXT: u32 = sycl::ext::intel::math::half2uint_rz(sycl::vec<uint16_t, 1>(u16).template as<sycl::vec<sycl::half, 1>>().x());
  // CHECK-INTEL-EXT: u16 = sycl::ext::intel::math::half2ushort_rz(sycl::vec<uint16_t, 1>(u16).template as<sycl::vec<sycl::half, 1>>().x());
  // CHECK-INTEL-EXT: u32 = sycl::ext::intel::math::half2ushort_rz(sycl::vec<uint16_t, 1>(u16).template as<sycl::vec<sycl::half, 1>>().x());
  asm volatile ("cvt.rzi.u64.f16 %0, %1;" : "=l"(u64) : "h"(u16));
  asm volatile ("cvt.rzi.u32.f16 %0, %1;" : "=r"(u32) : "h"(u16));
  asm volatile ("cvt.rzi.u16.f16 %0, %1;" : "=h"(u16) : "h"(u16));
  asm volatile ("cvt.rzi.u8.f16 %0, %1;" : "=r"(u32) : "h"(u16));

  // Test half to integer conversion with rmi
  // CHECK-INTEL-EXT: u64 = sycl::ext::intel::math::half2ull_rd(sycl::vec<uint16_t, 1>(u16).template as<sycl::vec<sycl::half, 1>>().x());
  // CHECK-INTEL-EXT: u32 = sycl::ext::intel::math::half2uint_rd(sycl::vec<uint16_t, 1>(u16).template as<sycl::vec<sycl::half, 1>>().x());
  // CHECK-INTEL-EXT: u16 = sycl::ext::intel::math::half2ushort_rd(sycl::vec<uint16_t, 1>(u16).template as<sycl::vec<sycl::half, 1>>().x());
  // CHECK-INTEL-EXT: u32 = sycl::ext::intel::math::half2ushort_rd(sycl::vec<uint16_t, 1>(u16).template as<sycl::vec<sycl::half, 1>>().x());
  asm volatile ("cvt.rmi.u64.f16 %0, %1;" : "=l"(u64) : "h"(u16));
  asm volatile ("cvt.rmi.u32.f16 %0, %1;" : "=r"(u32) : "h"(u16));
  asm volatile ("cvt.rmi.u16.f16 %0, %1;" : "=h"(u16) : "h"(u16));
  asm volatile ("cvt.rmi.u8.f16 %0, %1;" : "=r"(u32) : "h"(u16));

  // Test half to integer conversion with rpi
  // CHECK-INTEL-EXT: u64 = sycl::ext::intel::math::half2ull_ru(sycl::vec<uint16_t, 1>(u16).template as<sycl::vec<sycl::half, 1>>().x());
  // CHECK-INTEL-EXT: u32 = sycl::ext::intel::math::half2uint_ru(sycl::vec<uint16_t, 1>(u16).template as<sycl::vec<sycl::half, 1>>().x());
  // CHECK-INTEL-EXT: u16 = sycl::ext::intel::math::half2ushort_ru(sycl::vec<uint16_t, 1>(u16).template as<sycl::vec<sycl::half, 1>>().x());
  // CHECK-INTEL-EXT: u32 = sycl::ext::intel::math::half2ushort_ru(sycl::vec<uint16_t, 1>(u16).template as<sycl::vec<sycl::half, 1>>().x());
  asm volatile ("cvt.rpi.u64.f16 %0, %1;" : "=l"(u64) : "h"(u16));
  asm volatile ("cvt.rpi.u32.f16 %0, %1;" : "=r"(u32) : "h"(u16));
  asm volatile ("cvt.rpi.u16.f16 %0, %1;" : "=h"(u16) : "h"(u16));
  asm volatile ("cvt.rpi.u8.f16 %0, %1;" : "=r"(u32) : "h"(u16));

  // Test integer to half conversion with rn
  // CHECK-INTEL-EXT: u16 = sycl::vec<uint16_t, 1>(sycl::ext::intel::math::ull2half_rn(u64)).template as<sycl::vec<uint16_t, 1>>().x();
  // CHECK-INTEL-EXT: u16 = sycl::vec<uint16_t, 1>(sycl::ext::intel::math::uint2half_rn(u32)).template as<sycl::vec<uint16_t, 1>>().x();
  // CHECK-INTEL-EXT: u16 = sycl::vec<uint16_t, 1>(sycl::ext::intel::math::ushort2half_rn(u16)).template as<sycl::vec<uint16_t, 1>>().x();
  // CHECK-INTEL-EXT: u16 = sycl::vec<uint16_t, 1>(sycl::ext::intel::math::ushort2half_rn(u32)).template as<sycl::vec<uint16_t, 1>>().x();
  asm volatile ("cvt.rn.f16.u64 %0, %1;" : "=h"(u16) : "l"(u64));
  asm volatile ("cvt.rn.f16.u32 %0, %1;" : "=h"(u16) : "r"(u32));
  asm volatile ("cvt.rn.f16.u16 %0, %1;" : "=h"(u16) : "h"(u16));
  asm volatile ("cvt.rn.f16.u8 %0, %1;"  : "=h"(u16) : "r"(u32));

  // Test integer to half conversion with rz
  // CHECK-INTEL-EXT: u16 = sycl::vec<uint16_t, 1>(sycl::ext::intel::math::ull2half_rz(u64)).template as<sycl::vec<uint16_t, 1>>().x();
  // CHECK-INTEL-EXT: u16 = sycl::vec<uint16_t, 1>(sycl::ext::intel::math::uint2half_rz(u32)).template as<sycl::vec<uint16_t, 1>>().x();
  // CHECK-INTEL-EXT: u16 = sycl::vec<uint16_t, 1>(sycl::ext::intel::math::ushort2half_rz(u16)).template as<sycl::vec<uint16_t, 1>>().x();
  // CHECK-INTEL-EXT: u16 = sycl::vec<uint16_t, 1>(sycl::ext::intel::math::ushort2half_rz(u32)).template as<sycl::vec<uint16_t, 1>>().x();
  asm volatile ("cvt.rz.f16.u64 %0, %1;" : "=h"(u16) : "l"(u64));
  asm volatile ("cvt.rz.f16.u32 %0, %1;" : "=h"(u16) : "r"(u32));
  asm volatile ("cvt.rz.f16.u16 %0, %1;" : "=h"(u16) : "h"(u16));
  asm volatile ("cvt.rz.f16.u8 %0, %1;"  : "=h"(u16) : "r"(u32));

  // Test integer to half conversion with rm
  // CHECK-INTEL-EXT: u16 = sycl::vec<uint16_t, 1>(sycl::ext::intel::math::ull2half_rd(u64)).template as<sycl::vec<uint16_t, 1>>().x();
  // CHECK-INTEL-EXT: u16 = sycl::vec<uint16_t, 1>(sycl::ext::intel::math::uint2half_rd(u32)).template as<sycl::vec<uint16_t, 1>>().x();
  // CHECK-INTEL-EXT: u16 = sycl::vec<uint16_t, 1>(sycl::ext::intel::math::ushort2half_rd(u16)).template as<sycl::vec<uint16_t, 1>>().x();
  // CHECK-INTEL-EXT: u16 = sycl::vec<uint16_t, 1>(sycl::ext::intel::math::ushort2half_rd(u32)).template as<sycl::vec<uint16_t, 1>>().x();
  asm volatile ("cvt.rm.f16.u64 %0, %1;" : "=h"(u16) : "l"(u64));
  asm volatile ("cvt.rm.f16.u32 %0, %1;" : "=h"(u16) : "r"(u32));
  asm volatile ("cvt.rm.f16.u16 %0, %1;" : "=h"(u16) : "h"(u16));
  asm volatile ("cvt.rm.f16.u8 %0, %1;"  : "=h"(u16) : "r"(u32));

  // Test integer to half conversion with rp
  // CHECK-INTEL-EXT: u16 = sycl::vec<uint16_t, 1>(sycl::ext::intel::math::ull2half_ru(u64)).template as<sycl::vec<uint16_t, 1>>().x();
  // CHECK-INTEL-EXT: u16 = sycl::vec<uint16_t, 1>(sycl::ext::intel::math::uint2half_ru(u32)).template as<sycl::vec<uint16_t, 1>>().x();
  // CHECK-INTEL-EXT: u16 = sycl::vec<uint16_t, 1>(sycl::ext::intel::math::ushort2half_ru(u16)).template as<sycl::vec<uint16_t, 1>>().x();
  // CHECK-INTEL-EXT: u16 = sycl::vec<uint16_t, 1>(sycl::ext::intel::math::ushort2half_ru(u32)).template as<sycl::vec<uint16_t, 1>>().x();
  asm volatile ("cvt.rp.f16.u64 %0, %1;" : "=h"(u16) : "l"(u64));
  asm volatile ("cvt.rp.f16.u32 %0, %1;" : "=h"(u16) : "r"(u32));
  asm volatile ("cvt.rp.f16.u16 %0, %1;" : "=h"(u16) : "h"(u16));
  asm volatile ("cvt.rp.f16.u8 %0, %1;"  : "=h"(u16) : "r"(u32));

  // Test bfloat16 to integer conversion with rni
  // CHECK-INTEL-EXT: u64 = sycl::ext::intel::math::bfloat162ull_rn(sycl::vec<uint16_t, 1>(u16).template as<sycl::vec<sycl::ext::oneapi::bfloat16, 1>>().x());
  // CHECK-INTEL-EXT: u32 = sycl::ext::intel::math::bfloat162uint_rn(sycl::vec<uint16_t, 1>(u16).template as<sycl::vec<sycl::ext::oneapi::bfloat16, 1>>().x());
  // CHECK-INTEL-EXT: u16 = sycl::ext::intel::math::bfloat162ushort_rn(sycl::vec<uint16_t, 1>(u16).template as<sycl::vec<sycl::ext::oneapi::bfloat16, 1>>().x());
  asm volatile ("cvt.rni.u64.bf16 %0, %1;" : "=l"(u64) : "h"(u16));
  asm volatile ("cvt.rni.u32.bf16 %0, %1;" : "=r"(u32) : "h"(u16));
  asm volatile ("cvt.rni.u16.bf16 %0, %1;" : "=h"(u16) : "h"(u16));

  // Test bfloat16 to integer conversion with rzi
  // CHECK-INTEL-EXT: u64 = sycl::ext::intel::math::bfloat162ull_rz(sycl::vec<uint16_t, 1>(u16).template as<sycl::vec<sycl::ext::oneapi::bfloat16, 1>>().x());
  // CHECK-INTEL-EXT: u32 = sycl::ext::intel::math::bfloat162uint_rz(sycl::vec<uint16_t, 1>(u16).template as<sycl::vec<sycl::ext::oneapi::bfloat16, 1>>().x());
  // CHECK-INTEL-EXT: u16 = sycl::ext::intel::math::bfloat162ushort_rz(sycl::vec<uint16_t, 1>(u16).template as<sycl::vec<sycl::ext::oneapi::bfloat16, 1>>().x());
  asm volatile ("cvt.rzi.u64.bf16 %0, %1;" : "=l"(u64) : "h"(u16));
  asm volatile ("cvt.rzi.u32.bf16 %0, %1;" : "=r"(u32) : "h"(u16));
  asm volatile ("cvt.rzi.u16.bf16 %0, %1;" : "=h"(u16) : "h"(u16));

  // Test bfloat16 to integer conversion with rmi
  // CHECK-INTEL-EXT: u64 = sycl::ext::intel::math::bfloat162ull_rd(sycl::vec<uint16_t, 1>(u16).template as<sycl::vec<sycl::ext::oneapi::bfloat16, 1>>().x());
  // CHECK-INTEL-EXT: u32 = sycl::ext::intel::math::bfloat162uint_rd(sycl::vec<uint16_t, 1>(u16).template as<sycl::vec<sycl::ext::oneapi::bfloat16, 1>>().x());
  // CHECK-INTEL-EXT: u16 = sycl::ext::intel::math::bfloat162ushort_rd(sycl::vec<uint16_t, 1>(u16).template as<sycl::vec<sycl::ext::oneapi::bfloat16, 1>>().x());
  asm volatile ("cvt.rmi.u64.bf16 %0, %1;" : "=l"(u64) : "h"(u16));
  asm volatile ("cvt.rmi.u32.bf16 %0, %1;" : "=r"(u32) : "h"(u16));
  asm volatile ("cvt.rmi.u16.bf16 %0, %1;" : "=h"(u16) : "h"(u16));

  // Test bfloat16 to integer conversion with rpi
  // CHECK-INTEL-EXT: u64 = sycl::ext::intel::math::bfloat162ull_ru(sycl::vec<uint16_t, 1>(u16).template as<sycl::vec<sycl::ext::oneapi::bfloat16, 1>>().x());
  // CHECK-INTEL-EXT: u32 = sycl::ext::intel::math::bfloat162uint_ru(sycl::vec<uint16_t, 1>(u16).template as<sycl::vec<sycl::ext::oneapi::bfloat16, 1>>().x());
  // CHECK-INTEL-EXT: u16 = sycl::ext::intel::math::bfloat162ushort_ru(sycl::vec<uint16_t, 1>(u16).template as<sycl::vec<sycl::ext::oneapi::bfloat16, 1>>().x());
  asm volatile ("cvt.rpi.u64.bf16 %0, %1;" : "=l"(u64) : "h"(u16));
  asm volatile ("cvt.rpi.u32.bf16 %0, %1;" : "=r"(u32) : "h"(u16));
  asm volatile ("cvt.rpi.u16.bf16 %0, %1;" : "=h"(u16) : "h"(u16));

  // Test integer to bfloat16 conversion with rn
  // CHECK-INTEL-EXT: u16 = sycl::vec<uint16_t, 1>(sycl::ext::intel::math::ull2bfloat16_rn(u64)).template as<sycl::vec<uint16_t, 1>>().x();
  // CHECK-INTEL-EXT: u16 = sycl::vec<uint16_t, 1>(sycl::ext::intel::math::uint2bfloat16_rn(u32)).template as<sycl::vec<uint16_t, 1>>().x();
  // CHECK-INTEL-EXT: u16 = sycl::vec<uint16_t, 1>(sycl::ext::intel::math::ushort2bfloat16_rn(u16)).template as<sycl::vec<uint16_t, 1>>().x();
  // CHECK-INTEL-EXT: u16 = sycl::vec<uint16_t, 1>(sycl::ext::intel::math::float2bfloat16_rn(f32)).template as<sycl::vec<uint16_t, 1>>().x();
  asm volatile ("cvt.rn.bf16.u64 %0, %1;" : "=h"(u16) : "l"(u64));
  asm volatile ("cvt.rn.bf16.u32 %0, %1;" : "=h"(u16) : "r"(u32));
  asm volatile ("cvt.rn.bf16.u16 %0, %1;" : "=h"(u16) : "h"(u16));
  asm volatile ("cvt.rn.bf16.f32 %0, %1;" : "=h"(u16) : "f"(f32));

  // Test integer to bfloat16 conversion with rz
  // CHECK-INTEL-EXT: u16 = sycl::vec<uint16_t, 1>(sycl::ext::intel::math::ull2bfloat16_rz(u64)).template as<sycl::vec<uint16_t, 1>>().x();
  // CHECK-INTEL-EXT: u16 = sycl::vec<uint16_t, 1>(sycl::ext::intel::math::uint2bfloat16_rz(u32)).template as<sycl::vec<uint16_t, 1>>().x();
  // CHECK-INTEL-EXT: u16 = sycl::vec<uint16_t, 1>(sycl::ext::intel::math::ushort2bfloat16_rz(u16)).template as<sycl::vec<uint16_t, 1>>().x();
  asm volatile ("cvt.rz.bf16.u64 %0, %1;" : "=h"(u16) : "l"(u64));
  asm volatile ("cvt.rz.bf16.u32 %0, %1;" : "=h"(u16) : "r"(u32));
  asm volatile ("cvt.rz.bf16.u16 %0, %1;" : "=h"(u16) : "h"(u16));

  // Test integer to bfloat16 conversion with rm
  // CHECK-INTEL-EXT: u16 = sycl::vec<uint16_t, 1>(sycl::ext::intel::math::ull2bfloat16_rd(u64)).template as<sycl::vec<uint16_t, 1>>().x();
  // CHECK-INTEL-EXT: u16 = sycl::vec<uint16_t, 1>(sycl::ext::intel::math::uint2bfloat16_rd(u32)).template as<sycl::vec<uint16_t, 1>>().x();
  // CHECK-INTEL-EXT: u16 = sycl::vec<uint16_t, 1>(sycl::ext::intel::math::ushort2bfloat16_rd(u16)).template as<sycl::vec<uint16_t, 1>>().x();
  asm volatile ("cvt.rm.bf16.u64 %0, %1;" : "=h"(u16) : "l"(u64));
  asm volatile ("cvt.rm.bf16.u32 %0, %1;" : "=h"(u16) : "r"(u32));
  asm volatile ("cvt.rm.bf16.u16 %0, %1;" : "=h"(u16) : "h"(u16));

  // Test integer to bfloat16 conversion with rp
  // CHECK-INTEL-EXT: u16 = sycl::vec<uint16_t, 1>(sycl::ext::intel::math::ull2bfloat16_ru(u64)).template as<sycl::vec<uint16_t, 1>>().x();
  // CHECK-INTEL-EXT: u16 = sycl::vec<uint16_t, 1>(sycl::ext::intel::math::uint2bfloat16_ru(u32)).template as<sycl::vec<uint16_t, 1>>().x();
  // CHECK-INTEL-EXT: u16 = sycl::vec<uint16_t, 1>(sycl::ext::intel::math::ushort2bfloat16_ru(u16)).template as<sycl::vec<uint16_t, 1>>().x();
  asm volatile ("cvt.rp.bf16.u64 %0, %1;" : "=h"(u16) : "l"(u64));
  asm volatile ("cvt.rp.bf16.u32 %0, %1;" : "=h"(u16) : "r"(u32));
  asm volatile ("cvt.rp.bf16.u16 %0, %1;" : "=h"(u16) : "h"(u16));
}
#else
__global__ void cvt() {
  uint16_t u16;
  uint32_t u32;
  uint64_t u64;
  int16_t s16;
  int32_t s32;
  float f32;
  double f64;

  // Test half to integer conversion with rni
  // CHECK: u64 = sycl::vec<uint16_t, 1>(u16).template as<sycl::vec<sycl::half, 1>>().template convert<uint64_t, sycl::rounding_mode::rte>().x();
  // CHECK: u32 = sycl::vec<uint16_t, 1>(u16).template as<sycl::vec<sycl::half, 1>>().template convert<uint32_t, sycl::rounding_mode::rte>().x();
  // CHECK: u16 = sycl::vec<uint16_t, 1>(u16).template as<sycl::vec<sycl::half, 1>>().template convert<uint16_t, sycl::rounding_mode::rte>().x();
  // CHECK: u32 = sycl::vec<uint16_t, 1>(u16).template as<sycl::vec<sycl::half, 1>>().template convert<uint8_t, sycl::rounding_mode::rte>().x();
  asm volatile ("cvt.rni.u64.f16 %0, %1;" : "=l"(u64) : "h"(u16));
  asm volatile ("cvt.rni.u32.f16 %0, %1;" : "=r"(u32) : "h"(u16));
  asm volatile ("cvt.rni.u16.f16 %0, %1;" : "=h"(u16) : "h"(u16));
  asm volatile ("cvt.rni.u8.f16 %0, %1;" : "=r"(u32) : "h"(u16));

  // Test half to integer conversion with rzi
  // CHECK: u32 = sycl::vec<uint16_t, 1>(u16).template as<sycl::vec<sycl::half, 1>>().template convert<uint32_t, sycl::rounding_mode::rtz>().x();
  // CHECK: u16 = sycl::vec<uint16_t, 1>(u16).template as<sycl::vec<sycl::half, 1>>().template convert<uint16_t, sycl::rounding_mode::rtz>().x();
  // CHECK: u32 = sycl::vec<uint16_t, 1>(u16).template as<sycl::vec<sycl::half, 1>>().template convert<uint8_t, sycl::rounding_mode::rtz>().x();
  asm volatile ("cvt.rzi.u64.f16 %0, %1;" : "=r"(u64) : "h"(u16));
  asm volatile ("cvt.rzi.u32.f16 %0, %1;" : "=r"(u32) : "h"(u16));
  asm volatile ("cvt.rzi.u16.f16 %0, %1;" : "=h"(u16) : "h"(u16));
  asm volatile ("cvt.rzi.u8.f16 %0, %1;" : "=r"(u32) : "h"(u16));

  // Test half to integer conversion with rmi
  // CHECK: u64 = sycl::vec<uint16_t, 1>(u16).template as<sycl::vec<sycl::half, 1>>().template convert<uint64_t, sycl::rounding_mode::rtn>().x();
  // CHECK: u32 = sycl::vec<uint16_t, 1>(u16).template as<sycl::vec<sycl::half, 1>>().template convert<uint32_t, sycl::rounding_mode::rtn>().x();
  // CHECK: u16 = sycl::vec<uint16_t, 1>(u16).template as<sycl::vec<sycl::half, 1>>().template convert<uint16_t, sycl::rounding_mode::rtn>().x();
  // CHECK: u32 = sycl::vec<uint16_t, 1>(u16).template as<sycl::vec<sycl::half, 1>>().template convert<uint8_t, sycl::rounding_mode::rtn>().x();
  asm volatile ("cvt.rmi.u64.f16 %0, %1;" : "=l"(u64) : "h"(u16));
  asm volatile ("cvt.rmi.u32.f16 %0, %1;" : "=r"(u32) : "h"(u16));
  asm volatile ("cvt.rmi.u16.f16 %0, %1;" : "=h"(u16) : "h"(u16));
  asm volatile ("cvt.rmi.u8.f16 %0, %1;" : "=r"(u32) : "h"(u16));

  // Test half to integer conversion with rpi
  // CHECK: u64 = sycl::vec<uint16_t, 1>(u16).template as<sycl::vec<sycl::half, 1>>().template convert<uint64_t, sycl::rounding_mode::rtp>().x();
  // CHECK: u32 = sycl::vec<uint16_t, 1>(u16).template as<sycl::vec<sycl::half, 1>>().template convert<uint32_t, sycl::rounding_mode::rtp>().x();
  // CHECK: u16 = sycl::vec<uint16_t, 1>(u16).template as<sycl::vec<sycl::half, 1>>().template convert<uint16_t, sycl::rounding_mode::rtp>().x();
  // CHECK: u32 = sycl::vec<uint16_t, 1>(u16).template as<sycl::vec<sycl::half, 1>>().template convert<uint8_t, sycl::rounding_mode::rtp>().x();
  asm volatile ("cvt.rpi.u64.f16 %0, %1;" : "=l"(u64) : "h"(u16));
  asm volatile ("cvt.rpi.u32.f16 %0, %1;" : "=r"(u32) : "h"(u16));
  asm volatile ("cvt.rpi.u16.f16 %0, %1;" : "=h"(u16) : "h"(u16));
  asm volatile ("cvt.rpi.u8.f16 %0, %1;" : "=r"(u32) : "h"(u16));

  // Test integer to half conversion with rn
  // CHECK: u16 = sycl::vec<uint64_t, 1>(u64).template convert<sycl::half, sycl::rounding_mode::rte>().template as<sycl::vec<uint16_t, 1>>().x();
  // CHECK: u16 = sycl::vec<uint32_t, 1>(u32).template convert<sycl::half, sycl::rounding_mode::rte>().template as<sycl::vec<uint16_t, 1>>().x();
  // CHECK: u16 = sycl::vec<uint16_t, 1>(u16).template convert<sycl::half, sycl::rounding_mode::rte>().template as<sycl::vec<uint16_t, 1>>().x();
  // CHECK: u16 = sycl::vec<uint8_t, 1>(u32).template convert<sycl::half, sycl::rounding_mode::rte>().template as<sycl::vec<uint16_t, 1>>().x();
  asm volatile ("cvt.rn.f16.u64 %0, %1;" : "=h"(u16) : "l"(u64));
  asm volatile ("cvt.rn.f16.u32 %0, %1;" : "=h"(u16) : "r"(u32));
  asm volatile ("cvt.rn.f16.u16 %0, %1;" : "=h"(u16) : "h"(u16));
  asm volatile ("cvt.rn.f16.u8 %0, %1;"  : "=h"(u16) : "r"(u32));

  // Test integer to half conversion with rz
  // CHECK: u16 = sycl::vec<uint64_t, 1>(u64).template convert<sycl::half, sycl::rounding_mode::rtz>().template as<sycl::vec<uint16_t, 1>>().x();
  // CHECK: u16 = sycl::vec<uint32_t, 1>(u32).template convert<sycl::half, sycl::rounding_mode::rtz>().template as<sycl::vec<uint16_t, 1>>().x();
  // CHECK: u16 = sycl::vec<uint16_t, 1>(u16).template convert<sycl::half, sycl::rounding_mode::rtz>().template as<sycl::vec<uint16_t, 1>>().x();
  // CHECK: u16 = sycl::vec<uint8_t, 1>(u32).template convert<sycl::half, sycl::rounding_mode::rtz>().template as<sycl::vec<uint16_t, 1>>().x();
  asm volatile ("cvt.rz.f16.u64 %0, %1;" : "=h"(u16) : "l"(u64));
  asm volatile ("cvt.rz.f16.u32 %0, %1;" : "=h"(u16) : "r"(u32));
  asm volatile ("cvt.rz.f16.u16 %0, %1;" : "=h"(u16) : "h"(u16));
  asm volatile ("cvt.rz.f16.u8 %0, %1;"  : "=h"(u16) : "r"(u32));

  // Test integer to half conversion with rm
  // CHECK: u16 = sycl::vec<uint64_t, 1>(u64).template convert<sycl::half, sycl::rounding_mode::rtn>().template as<sycl::vec<uint16_t, 1>>().x();
  // CHECK: u16 = sycl::vec<uint32_t, 1>(u32).template convert<sycl::half, sycl::rounding_mode::rtn>().template as<sycl::vec<uint16_t, 1>>().x();
  // CHECK: u16 = sycl::vec<uint16_t, 1>(u16).template convert<sycl::half, sycl::rounding_mode::rtn>().template as<sycl::vec<uint16_t, 1>>().x();
  // CHECK: u16 = sycl::vec<uint8_t, 1>(u32).template convert<sycl::half, sycl::rounding_mode::rtn>().template as<sycl::vec<uint16_t, 1>>().x();
  asm volatile ("cvt.rm.f16.u64 %0, %1;" : "=h"(u16) : "l"(u64));
  asm volatile ("cvt.rm.f16.u32 %0, %1;" : "=h"(u16) : "r"(u32));
  asm volatile ("cvt.rm.f16.u16 %0, %1;" : "=h"(u16) : "h"(u16));
  asm volatile ("cvt.rm.f16.u8 %0, %1;"  : "=h"(u16) : "r"(u32));

  // Test integer to half conversion with rp
  // CHECK: u16 = sycl::vec<uint64_t, 1>(u64).template convert<sycl::half, sycl::rounding_mode::rtp>().template as<sycl::vec<uint16_t, 1>>().x();
  // CHECK: u16 = sycl::vec<uint32_t, 1>(u32).template convert<sycl::half, sycl::rounding_mode::rtp>().template as<sycl::vec<uint16_t, 1>>().x();
  // CHECK: u16 = sycl::vec<uint16_t, 1>(u16).template convert<sycl::half, sycl::rounding_mode::rtp>().template as<sycl::vec<uint16_t, 1>>().x();
  // CHECK: u16 = sycl::vec<uint8_t, 1>(u32).template convert<sycl::half, sycl::rounding_mode::rtp>().template as<sycl::vec<uint16_t, 1>>().x();
  asm volatile ("cvt.rp.f16.u64 %0, %1;" : "=h"(u16) : "l"(u64));
  asm volatile ("cvt.rp.f16.u32 %0, %1;" : "=h"(u16) : "r"(u32));
  asm volatile ("cvt.rp.f16.u16 %0, %1;" : "=h"(u16) : "h"(u16));
  asm volatile ("cvt.rp.f16.u8 %0, %1;"  : "=h"(u16) : "r"(u32));

  // CHECK: s16 = sycl::vec<float, 1>(f32).template convert<int16_t, sycl::rounding_mode::rte>().x();
  // CHECK: s16 = sycl::vec<double, 1>(f64).template convert<int16_t, sycl::rounding_mode::rte>().x();
  // CHECK: s32 = sycl::vec<float, 1>(f32).template convert<int8_t, sycl::rounding_mode::rte>().x();
  // CHECK: s32 = sycl::vec<double, 1>(f64).template convert<int8_t, sycl::rounding_mode::rte>().x();
  // CHECK: u16 = sycl::vec<float, 1>(f32).template convert<uint16_t, sycl::rounding_mode::rte>().x();
  // CHECK: u16 = sycl::vec<double, 1>(f64).template convert<uint16_t, sycl::rounding_mode::rte>().x();
  // CHECK: u32 = sycl::vec<float, 1>(f32).template convert<uint8_t, sycl::rounding_mode::rte>().x();
  // CHECK: u32 = sycl::vec<double, 1>(f64).template convert<uint8_t, sycl::rounding_mode::rte>().x();
  asm volatile("cvt.rni.sat.s16.f32 %0, %1;" : "=h"(s16) : "f"(f32));
  asm volatile("cvt.rni.sat.s16.f64 %0, %1;" : "=h"(s16) : "d"(f64));
  asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(s32) : "f"(f32));
  asm volatile("cvt.rni.sat.s8.f64 %0, %1;" : "=r"(s32) : "d"(f64));
  asm volatile("cvt.rni.sat.u16.f32 %0, %1;" : "=h"(u16) : "f"(f32));
  asm volatile("cvt.rni.sat.u16.f64 %0, %1;" : "=h"(u16) : "d"(f64));
  asm volatile("cvt.rni.sat.u8.f32 %0, %1;" : "=r"(u32) : "f"(f32));
  asm volatile("cvt.rni.sat.u8.f64 %0, %1;" : "=r"(u32) : "d"(f64));
  
  // CHECK: s16 = sycl::vec<float, 1>(f32).template convert<int16_t, sycl::rounding_mode::rtn>().x();
  // CHECK: s16 = sycl::vec<double, 1>(f64).template convert<int16_t, sycl::rounding_mode::rtn>().x();
  // CHECK: s32 = sycl::vec<float, 1>(f32).template convert<int8_t, sycl::rounding_mode::rtn>().x();
  // CHECK: s32 = sycl::vec<double, 1>(f64).template convert<int8_t, sycl::rounding_mode::rtn>().x();
  // CHECK: u16 = sycl::vec<float, 1>(f32).template convert<uint16_t, sycl::rounding_mode::rtn>().x();
  // CHECK: u16 = sycl::vec<double, 1>(f64).template convert<uint16_t, sycl::rounding_mode::rtn>().x();
  // CHECK: u32 = sycl::vec<float, 1>(f32).template convert<uint8_t, sycl::rounding_mode::rtn>().x();
  // CHECK: u32 = sycl::vec<double, 1>(f64).template convert<uint8_t, sycl::rounding_mode::rtn>().x();
  asm volatile("cvt.rmi.sat.s16.f32 %0, %1;" : "=h"(s16) : "f"(f32));
  asm volatile("cvt.rmi.sat.s16.f64 %0, %1;" : "=h"(s16) : "d"(f64));
  asm volatile("cvt.rmi.sat.s8.f32 %0, %1;" : "=r"(s32) : "f"(f32));
  asm volatile("cvt.rmi.sat.s8.f64 %0, %1;" : "=r"(s32) : "d"(f64));
  asm volatile("cvt.rmi.sat.u16.f32 %0, %1;" : "=h"(u16) : "f"(f32));
  asm volatile("cvt.rmi.sat.u16.f64 %0, %1;" : "=h"(u16) : "d"(f64));
  asm volatile("cvt.rmi.sat.u8.f32 %0, %1;" : "=r"(u32) : "f"(f32));
  asm volatile("cvt.rmi.sat.u8.f64 %0, %1;" : "=r"(u32) : "d"(f64));

  // CHECK: s16 = sycl::vec<float, 1>(f32).template convert<int16_t, sycl::rounding_mode::rtz>().x();
  // CHECK: s16 = sycl::vec<double, 1>(f64).template convert<int16_t, sycl::rounding_mode::rtz>().x();
  // CHECK: s32 = sycl::vec<float, 1>(f32).template convert<int8_t, sycl::rounding_mode::rtz>().x();
  // CHECK: s32 = sycl::vec<double, 1>(f64).template convert<int8_t, sycl::rounding_mode::rtz>().x();
  // CHECK: u16 = sycl::vec<float, 1>(f32).template convert<uint16_t, sycl::rounding_mode::rtz>().x();
  // CHECK: u16 = sycl::vec<double, 1>(f64).template convert<uint16_t, sycl::rounding_mode::rtz>().x();
  // CHECK: u32 = sycl::vec<float, 1>(f32).template convert<uint8_t, sycl::rounding_mode::rtz>().x();
  // CHECK: u32 = sycl::vec<double, 1>(f64).template convert<uint8_t, sycl::rounding_mode::rtz>().x();
  asm volatile("cvt.rzi.sat.s16.f32 %0, %1;" : "=h"(s16) : "f"(f32));
  asm volatile("cvt.rzi.sat.s16.f64 %0, %1;" : "=h"(s16) : "d"(f64));
  asm volatile("cvt.rzi.sat.s8.f32 %0, %1;" : "=r"(s32) : "f"(f32));
  asm volatile("cvt.rzi.sat.s8.f64 %0, %1;" : "=r"(s32) : "d"(f64));
  asm volatile("cvt.rzi.sat.u16.f32 %0, %1;" : "=h"(u16) : "f"(f32));
  asm volatile("cvt.rzi.sat.u16.f64 %0, %1;" : "=h"(u16) : "d"(f64));
  asm volatile("cvt.rzi.sat.u8.f32 %0, %1;" : "=r"(u32) : "f"(f32));
  asm volatile("cvt.rzi.sat.u8.f64 %0, %1;" : "=r"(u32) : "d"(f64));

  // CHECK: s16 = sycl::vec<float, 1>(f32).template convert<int16_t, sycl::rounding_mode::rtp>().x();
  // CHECK: s16 = sycl::vec<double, 1>(f64).template convert<int16_t, sycl::rounding_mode::rtp>().x();
  // CHECK: s32 = sycl::vec<float, 1>(f32).template convert<int8_t, sycl::rounding_mode::rtp>().x();
  // CHECK: s32 = sycl::vec<double, 1>(f64).template convert<int8_t, sycl::rounding_mode::rtp>().x();
  // CHECK: u16 = sycl::vec<float, 1>(f32).template convert<uint16_t, sycl::rounding_mode::rtp>().x();
  // CHECK: u16 = sycl::vec<double, 1>(f64).template convert<uint16_t, sycl::rounding_mode::rtp>().x();
  // CHECK: u32 = sycl::vec<float, 1>(f32).template convert<uint8_t, sycl::rounding_mode::rtp>().x();
  // CHECK: u32 = sycl::vec<double, 1>(f64).template convert<uint8_t, sycl::rounding_mode::rtp>().x();
  asm volatile("cvt.rpi.sat.s16.f32 %0, %1;" : "=h"(s16) : "f"(f32));
  asm volatile("cvt.rpi.sat.s16.f64 %0, %1;" : "=h"(s16) : "d"(f64));
  asm volatile("cvt.rpi.sat.s8.f32 %0, %1;" : "=r"(s32) : "f"(f32));
  asm volatile("cvt.rpi.sat.s8.f64 %0, %1;" : "=r"(s32) : "d"(f64));
  asm volatile("cvt.rpi.sat.u16.f32 %0, %1;" : "=h"(u16) : "f"(f32));
  asm volatile("cvt.rpi.sat.u16.f64 %0, %1;" : "=h"(u16) : "d"(f64));
  asm volatile("cvt.rpi.sat.u8.f32 %0, %1;" : "=r"(u32) : "f"(f32));
  asm volatile("cvt.rpi.sat.u8.f64 %0, %1;" : "=r"(u32) : "d"(f64));

  // Test integer to integer conversion with rni
  // CHECK: s16 = sycl::vec<int32_t, 1>(s32).template convert<int16_t, sycl::rounding_mode::rte>().x();
  // CHECK: s16 = sycl::vec<int64_t, 1>(u64).template convert<int16_t, sycl::rounding_mode::rte>().x();
  // CHECK: s32 = sycl::vec<int16_t, 1>(s16).template convert<int32_t, sycl::rounding_mode::rte>().x();
  // CHECK: s32 = sycl::vec<int64_t, 1>(u64).template convert<int32_t, sycl::rounding_mode::rte>().x();
  // CHECK: u64 = sycl::vec<int16_t, 1>(s16).template convert<int64_t, sycl::rounding_mode::rte>().x();
  // CHECK: u64 = sycl::vec<int32_t, 1>(s32).template convert<int64_t, sycl::rounding_mode::rte>().x();
  asm volatile ("cvt.rni.s16.s32 %0, %1;" : "=h"(s16) : "r"(s32));
  asm volatile ("cvt.rni.s16.s64 %0, %1;" : "=h"(s16) : "l"(u64));
  asm volatile ("cvt.rni.s32.s16 %0, %1;" : "=r"(s32) : "h"(s16));
  asm volatile ("cvt.rni.s32.s64 %0, %1;" : "=r"(s32) : "l"(u64));
  asm volatile ("cvt.rni.s64.s16 %0, %1;" : "=l"(u64) : "h"(s16));
  asm volatile ("cvt.rni.s64.s32 %0, %1;" : "=l"(u64) : "r"(s32));

  // Test integer to integer conversion with rzi
  // CHECK: s16 = sycl::vec<int32_t, 1>(s32).template convert<int16_t, sycl::rounding_mode::rtz>().x();
  // CHECK: s16 = sycl::vec<int64_t, 1>(u64).template convert<int16_t, sycl::rounding_mode::rtz>().x();
  // CHECK: s32 = sycl::vec<int16_t, 1>(s16).template convert<int32_t, sycl::rounding_mode::rtz>().x();
  // CHECK: s32 = sycl::vec<int64_t, 1>(u64).template convert<int32_t, sycl::rounding_mode::rtz>().x();
  // CHECK: u64 = sycl::vec<int16_t, 1>(s16).template convert<int64_t, sycl::rounding_mode::rtz>().x();
  // CHECK: u64 = sycl::vec<int32_t, 1>(s32).template convert<int64_t, sycl::rounding_mode::rtz>().x();
  asm volatile ("cvt.rzi.s16.s32 %0, %1;" : "=h"(s16) : "r"(s32));
  asm volatile ("cvt.rzi.s16.s64 %0, %1;" : "=h"(s16) : "l"(u64));
  asm volatile ("cvt.rzi.s32.s16 %0, %1;" : "=r"(s32) : "h"(s16));
  asm volatile ("cvt.rzi.s32.s64 %0, %1;" : "=r"(s32) : "l"(u64));
  asm volatile ("cvt.rzi.s64.s16 %0, %1;" : "=l"(u64) : "h"(s16));
  asm volatile ("cvt.rzi.s64.s32 %0, %1;" : "=l"(u64) : "r"(s32));

  // Test integer to integer conversion with rmi
  // CHECK: s16 = sycl::vec<int32_t, 1>(s32).template convert<int16_t, sycl::rounding_mode::rtn>().x();
  // CHECK: s16 = sycl::vec<int64_t, 1>(u64).template convert<int16_t, sycl::rounding_mode::rtn>().x();
  // CHECK: s32 = sycl::vec<int16_t, 1>(s16).template convert<int32_t, sycl::rounding_mode::rtn>().x();
  // CHECK: s32 = sycl::vec<int64_t, 1>(u64).template convert<int32_t, sycl::rounding_mode::rtn>().x();
  // CHECK: u64 = sycl::vec<int16_t, 1>(s16).template convert<int64_t, sycl::rounding_mode::rtn>().x();
  // CHECK: u64 = sycl::vec<int32_t, 1>(s32).template convert<int64_t, sycl::rounding_mode::rtn>().x();
  asm volatile ("cvt.rmi.s16.s32 %0, %1;" : "=h"(s16) : "r"(s32));
  asm volatile ("cvt.rmi.s16.s64 %0, %1;" : "=h"(s16) : "l"(u64));
  asm volatile ("cvt.rmi.s32.s16 %0, %1;" : "=r"(s32) : "h"(s16));
  asm volatile ("cvt.rmi.s32.s64 %0, %1;" : "=r"(s32) : "l"(u64));
  asm volatile ("cvt.rmi.s64.s16 %0, %1;" : "=l"(u64) : "h"(s16));
  asm volatile ("cvt.rmi.s64.s32 %0, %1;" : "=l"(u64) : "r"(s32));

  // Test integer to integer conversion with rpi
  // CHECK: s16 = sycl::vec<int32_t, 1>(s32).template convert<int16_t, sycl::rounding_mode::rtp>().x();
  // CHECK: s16 = sycl::vec<int64_t, 1>(u64).template convert<int16_t, sycl::rounding_mode::rtp>().x();
  // CHECK: s32 = sycl::vec<int16_t, 1>(s16).template convert<int32_t, sycl::rounding_mode::rtp>().x();
  // CHECK: s32 = sycl::vec<int64_t, 1>(u64).template convert<int32_t, sycl::rounding_mode::rtp>().x();
  // CHECK: u64 = sycl::vec<int16_t, 1>(s16).template convert<int64_t, sycl::rounding_mode::rtp>().x();
  // CHECK: u64 = sycl::vec<int32_t, 1>(s32).template convert<int64_t, sycl::rounding_mode::rtp>().x();
  asm volatile ("cvt.rpi.s16.s32 %0, %1;" : "=h"(s16) : "r"(s32));
  asm volatile ("cvt.rpi.s16.s64 %0, %1;" : "=h"(s16) : "l"(u64));
  asm volatile ("cvt.rpi.s32.s16 %0, %1;" : "=r"(s32) : "h"(s16));
  asm volatile ("cvt.rpi.s32.s64 %0, %1;" : "=r"(s32) : "l"(u64));
  asm volatile ("cvt.rpi.s64.s16 %0, %1;" : "=l"(u64) : "h"(s16));
  asm volatile ("cvt.rpi.s64.s32 %0, %1;" : "=l"(u64) : "r"(s32));

  // Test integer to integer conversion without rounding modifier
  // CHECK: s16 = static_cast<int16_t>(s32);
  // CHECK: s16 = static_cast<int16_t>(u64);
  // CHECK: s32 = static_cast<int32_t>(s16);
  // CHECK: s32 = static_cast<int32_t>(u64);
  // CHECK: u64 = static_cast<int64_t>(s16);
  // CHECK: u64 = static_cast<int64_t>(s32);
  asm volatile ("cvt.s16.s32 %0, %1;" : "=h"(s16) : "r"(s32));
  asm volatile ("cvt.s16.s64 %0, %1;" : "=h"(s16) : "l"(u64));
  asm volatile ("cvt.s32.s16 %0, %1;" : "=r"(s32) : "h"(s16));
  asm volatile ("cvt.s32.s64 %0, %1;" : "=r"(s32) : "l"(u64));
  asm volatile ("cvt.s64.s16 %0, %1;" : "=l"(u64) : "h"(s16));
  asm volatile ("cvt.s64.s32 %0, %1;" : "=l"(u64) : "r"(s32));
}
#endif

// clang-format on
