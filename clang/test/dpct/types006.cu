// RUN: dpct -out-root %T/types006 %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/types006/types006.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/types006/types006.dp.cpp -o %T/types006/types006.dp.o %}

// CHECK: #include <oneapi/dpl/execution>
// CHECK-NEXT:#include <oneapi/dpl/algorithm>
// CHECK-NEXT:#include <sycl/sycl.hpp>
// CHECK-NEXT:#include <dpct/dpct.hpp>
// CHECK-NEXT:#include <dpct/rng_utils.hpp>
// CHECK-NEXT:#include <dpct/dpl_utils.hpp>
#include <curand_kernel.h>
#include <cub/cub.cuh>

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

// CHECK:void foo(){
// CHECK-NEXT:  oneapi::dpl::reverse_iterator<int *> d_tmp(0);
// CHECK-NEXT:}
void foo(){
  thrust::reverse_iterator<int *> d_tmp(0);
}
