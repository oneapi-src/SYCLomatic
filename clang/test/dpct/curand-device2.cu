// RUN: dpct --format-range=none --usm-level=none --out-root %T/curand-device2 %s  --cuda-include-path="%cuda-path/include" --extra-arg="-std=c++14"
// RUN: FileCheck --input-file %T/curand-device2/curand-device2.dp.cpp --match-full-lines %s

#include "curand_kernel.h"


__global__ void kernel1() {
  unsigned int u;
  uint4 u4;
  float f;
  float2 f2;
  float4 f4;
  double d;
  double2 d2;
  double4 d4;

  //CHECK:dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> rng1;
  //CHECK-NEXT:rng1 = dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>>(1, {3, 2 * 4});
  //CHECK-NEXT:u = rng1.generate<oneapi::mkl::rng::device::bits<std::uint32_t>, 1>();
  curandStatePhilox4_32_10_t rng1;
  curand_init(1, 2, 3, &rng1);
  u = curand(&rng1);

  //CHECK:dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> rng2;
  //CHECK-NEXT:rng2 = dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>>(1, {3, 2 * 4});
  //CHECK-NEXT:u4 = rng2.generate<oneapi::mkl::rng::device::bits<std::uint32_t>, 4>();
  curandStatePhilox4_32_10_t rng2;
  curand_init(1, 2, 3, &rng2);
  u4 = curand4(&rng2);

  //CHECK:dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> rng3;
  //CHECK-NEXT:rng3 = dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>>(1, {3, 2 * 4});
  //CHECK-NEXT:f = rng3.generate<oneapi::mkl::rng::device::gaussian<float>, 1>();
  curandStatePhilox4_32_10_t rng3;
  curand_init(1, 2, 3, &rng3);
  f = curand_normal(&rng3);

  //CHECK:dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> rng4;
  //CHECK-NEXT:rng4 = dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>>(1, {3, 2 * 4});
  //CHECK-NEXT:f2 = rng4.generate<oneapi::mkl::rng::device::gaussian<float>, 2>();
  curandStatePhilox4_32_10_t rng4;
  curand_init(1, 2, 3, &rng4);
  f2 = curand_normal2(&rng4);

  //CHECK:dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> rng5;
  //CHECK-NEXT:rng5 = dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>>(1, {3, 2 * 4});
  //CHECK-NEXT:d2 = rng5.generate<oneapi::mkl::rng::device::gaussian<double>, 2>();
  curandStatePhilox4_32_10_t rng5;
  curand_init(1, 2, 3, &rng5);
  d2 = curand_normal2_double(&rng5);

  //CHECK:dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> rng6;
  //CHECK-NEXT:rng6 = dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>>(1, {3, 2 * 4});
  //CHECK-NEXT:f4 = rng6.generate<oneapi::mkl::rng::device::gaussian<float>, 4>();
  curandStatePhilox4_32_10_t rng6;
  curand_init(1, 2, 3, &rng6);
  f4 = curand_normal4(&rng6);

  //CHECK:dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> rng7;
  //CHECK-NEXT:rng7 = dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>>(1, {3, 2 * 4});
  //CHECK-NEXT:d = rng7.generate<oneapi::mkl::rng::device::gaussian<double>, 1>();
  curandStatePhilox4_32_10_t rng7;
  curand_init(1, 2, 3, &rng7);
  d = curand_normal_double(&rng7);

  //CHECK:dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> rng8;
  //CHECK-NEXT:rng8 = dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>>(1, {3, 2 * 4});
  //CHECK-NEXT:f = rng8.generate<oneapi::mkl::rng::device::lognormal<float>, 1>(3, 7);
  curandStatePhilox4_32_10_t rng8;
  curand_init(1, 2, 3, &rng8);
  f = curand_log_normal(&rng8, 3, 7);

  //CHECK:dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> rng9;
  //CHECK-NEXT:rng9 = dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>>(1, {3, 2 * 4});
  //CHECK-NEXT:f2 = rng9.generate<oneapi::mkl::rng::device::lognormal<float>, 2>(3, 7);
  curandStatePhilox4_32_10_t rng9;
  curand_init(1, 2, 3, &rng9);
  f2 = curand_log_normal2(&rng9, 3, 7);

  //CHECK:dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> rng10;
  //CHECK-NEXT:rng10 = dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>>(1, {3, 2 * 4});
  //CHECK-NEXT:d2 = rng10.generate<oneapi::mkl::rng::device::lognormal<double>, 2>(3, 7);
  curandStatePhilox4_32_10_t rng10;
  curand_init(1, 2, 3, &rng10);
  d2 = curand_log_normal2_double(&rng10, 3, 7);

  //CHECK:dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> rng11;
  //CHECK-NEXT:rng11 = dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>>(1, {3, 2 * 4});
  //CHECK-NEXT:f4 = rng11.generate<oneapi::mkl::rng::device::lognormal<float>, 4>(3, 7);
  curandStatePhilox4_32_10_t rng11;
  curand_init(1, 2, 3, &rng11);
  f4 = curand_log_normal4(&rng11, 3, 7);

  //CHECK:dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> rng12;
  //CHECK-NEXT:rng12 = dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>>(1, {3, 2 * 4});
  //CHECK-NEXT:d = rng12.generate<oneapi::mkl::rng::device::lognormal<double>, 1>(3, 7);
  curandStatePhilox4_32_10_t rng12;
  curand_init(1, 2, 3, &rng12);
  d = curand_log_normal_double(&rng12, 3, 7);

  //CHECK:dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> rng13;
  //CHECK-NEXT:rng13 = dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>>(1, {3, 2 * 4});
  //CHECK-NEXT:f = rng13.generate<oneapi::mkl::rng::device::uniform<float>, 1>();
  curandStatePhilox4_32_10_t rng13;
  curand_init(1, 2, 3, &rng13);
  f = curand_uniform(&rng13);

  //CHECK:dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> rng14;
  //CHECK-NEXT:rng14 = dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>>(1, {3, 2 * 4});
  //CHECK-NEXT:d2 = rng14.generate<oneapi::mkl::rng::device::uniform<double>, 2>();
  curandStatePhilox4_32_10_t rng14;
  curand_init(1, 2, 3, &rng14);
  d2 = curand_uniform2_double(&rng14);

  //CHECK:dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> rng15;
  //CHECK-NEXT:rng15 = dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>>(1, {3, 2 * 4});
  //CHECK-NEXT:f4 = rng15.generate<oneapi::mkl::rng::device::uniform<float>, 4>();
  curandStatePhilox4_32_10_t rng15;
  curand_init(1, 2, 3, &rng15);
  f4 = curand_uniform4(&rng15);

  //CHECK:dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> rng16;
  //CHECK-NEXT:rng16 = dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>>(1, {3, 2 * 4});
  //CHECK-NEXT:d = rng16.generate<oneapi::mkl::rng::device::uniform<double>, 1>();
  curandStatePhilox4_32_10_t rng16;
  curand_init(1, 2, 3, &rng16);
  d = curand_uniform_double(&rng16);

  //CHECK:dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> rng17;
  //CHECK-NEXT:rng17 = dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>>(1, {3, 2 * 4});
  //CHECK-NEXT:u = rng17.generate<oneapi::mkl::rng::device::poisson<std::uint32_t>, 1>(3);
  curandStatePhilox4_32_10_t rng17;
  curand_init(1, 2, 3, &rng17);
  u = curand_poisson(&rng17, 3);

  //CHECK:dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> rng18;
  //CHECK-NEXT:rng18 = dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>>(1, {3, 2 * 4});
  //CHECK-NEXT:u4 = rng18.generate<oneapi::mkl::rng::device::poisson<std::uint32_t>, 4>(3);
  curandStatePhilox4_32_10_t rng18;
  curand_init(1, 2, 3, &rng18);
  u4 = curand_poisson4(&rng18, 3);

  //CHECK:dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> rng19;
  //CHECK-NEXT:rng19 = dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>>(1, {3, 2 * 4});
  //CHECK-NEXT:d4 = rng19.generate<oneapi::mkl::rng::device::uniform<double>, 4>();
  curandStatePhilox4_32_10_t rng19;
  curand_init(1, 2, 3, &rng19);
  d4 = curand_uniform4_double(&rng19);

  //CHECK:dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> rng20;
  //CHECK-NEXT:rng20 = dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>>(1, {3, 2 * 4});
  //CHECK-NEXT:d4 = rng20.generate<oneapi::mkl::rng::device::gaussian<double>, 4>();
  curandStatePhilox4_32_10_t rng20;
  curand_init(1, 2, 3, &rng20);
  d4 = curand_normal4_double(&rng20);

  //CHECK:dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> rng21;
  //CHECK-NEXT:rng21 = dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>>(1, {3, 2 * 4});
  //CHECK-NEXT:d4 = rng21.generate<oneapi::mkl::rng::device::lognormal<double>, 4>(3, 7);
  curandStatePhilox4_32_10_t rng21;
  curand_init(1, 2, 3, &rng21);
  d4 = curand_log_normal4_double(&rng21, 3, 7);
}

__global__ void kernel2() {
//CHECK:dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> rng1;
//CHECK-NEXT:dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> rng2;
//CHECK-NEXT:rng1 = dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>>(11, {1234, 1 * 4});
//CHECK-NEXT:rng2 = dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>>(22, {4321, 2 * 4});
//CHECK-NEXT:float x = rng1.generate<oneapi::mkl::rng::device::uniform<float>, 1>();
//CHECK-NEXT:sycl::float2 y = rng2.generate<oneapi::mkl::rng::device::gaussian<float>, 2>();
  curandStatePhilox4_32_10_t rng1;
  curandStatePhilox4_32_10_t rng2;
  curand_init(11, 1, 1234, &rng1);
  curand_init(22, 2, 4321, &rng2);
  float x = curand_uniform(&rng1);
  float2 y = curand_normal2(&rng2);
}

__global__ void kernel3() {
  curandStateMRG32k3a_t rng1;
  curandStatePhilox4_32_10_t rng2;
  curandStateXORWOW_t rng3;

  curand_init(1, 2, 3, &rng1);
  curand_init(1, 2, 3, &rng2);
  curand_init(1, 2, 3, &rng3);

  //CHECK:oneapi::mkl::rng::device::skip_ahead(rng1.get_engine(), 1);
  //CHECK-NEXT:oneapi::mkl::rng::device::skip_ahead(rng2.get_engine(), 2);
  //CHECK-NEXT:oneapi::mkl::rng::device::skip_ahead(rng3.get_engine(), 3);
  skipahead(1, &rng1);
  skipahead(2, &rng2);
  skipahead(3, &rng3);

  //CHECK:oneapi::mkl::rng::device::skip_ahead(rng1.get_engine(), {0, 1 * (std::uint64_t(1) << 63)});
  //CHECK-NEXT:oneapi::mkl::rng::device::skip_ahead(rng2.get_engine(), {0, static_cast<std::uint64_t>(2 * 4)});
  //CHECK-NEXT:oneapi::mkl::rng::device::skip_ahead(rng3.get_engine(), {0, static_cast<std::uint64_t>(3 * 8)});
  skipahead_sequence(1, &rng1);
  skipahead_sequence(2, &rng2);
  skipahead_sequence(3, &rng3);

  //CHECK:oneapi::mkl::rng::device::skip_ahead(rng1.get_engine(), {0, static_cast<std::uint64_t>(1 * 8)});
  skipahead_subsequence(1, &rng1);

  curand_uniform(&rng1);
  curand_uniform(&rng2);
  curand_uniform(&rng3);
}

__global__ void type_test() {
  //CHECK:dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>> rng1;
  curandStateXORWOW_t rng1;
  //CHECK:dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>> rng2;
  curandStateXORWOW rng2;
  //CHECK:dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>> rng3;
  curandState_t rng3;
  //CHECK:dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>> rng4;
  curandState rng4;
  //CHECK:dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> rng5;
  curandStatePhilox4_32_10_t rng5;
  //CHECK:dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> rng6;
  curandStatePhilox4_32_10 rng6;
  //CHECK:dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mrg32k3a<1>> rng7;
  curandStateMRG32k3a_t rng7;
  //CHECK:dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mrg32k3a<1>> rng8;
  curandStateMRG32k3a rng8;
}

int main() {
  kernel1<<<1,1>>>();
  kernel2<<<1,1>>>();
  kernel3<<<1,1>>>();
  return 0;
}

__global__ void kernel4() {
  curandStateMRG32k3a_t rng;
  //CHECK:rng = dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mrg32k3a<1>>(1, {4, static_cast<std::uint64_t>((2 + 3) * 8)});
  //CHECK-NEXT:oneapi::mkl::rng::device::skip_ahead(rng.get_engine(), {0, (2 + 3) * (std::uint64_t(1) << 63)});
  //CHECK-NEXT:oneapi::mkl::rng::device::skip_ahead(rng.get_engine(), {0, static_cast<std::uint64_t>((2 + 3) * 8)});
  curand_init(1, 2 + 3, 4, &rng);
  skipahead_sequence(2 + 3, &rng);
  skipahead_subsequence(2 + 3, &rng);
}

