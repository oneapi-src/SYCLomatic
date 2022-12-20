// RUN: dpct -out-root %T/types006 %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/types006/types006.dp.cpp

#include <curand_kernel.h>

void foo() {
// CHECK: dpct::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<1>> bar;
// CHECK-NEXT: auto lambda = [&bar] () {
// CHECK-NEXT:   return bar.generate<oneapi::mkl::rng::device::uniform<float>, 1>();
// CHECK-NEXT: };
  curandStatePhilox4_32_10_t bar;
  auto lambda = [&bar] __device__ () {
    return curand_uniform(&bar);
  };
}
