// RUN: dpct --format-range=none -extra-arg-before=-std=c++14 -out-root %T/curand-device-different-vec-size %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/curand-device-different-vec-size/curand-device-different-vec-size.dp.cpp --match-full-lines %s

#include <cuda.h>
#include <curand_kernel.h>
#include <cstdio>

const int WARP_SIZE = 32;
const int NBLOCKS = 640;
const int ITERATIONS = 1000000;


__global__ void picount(int *totals) {
  __shared__ int counter[WARP_SIZE];
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  // CHECK: DPCT1050:{{[0-9]+}}: The template argument of the RNG engine could not be deduced. You need to update this code.
  // CHECK: oneapi::mkl::rng::device::philox4x32x10<dpct_placeholder/*Fix the vec_size manually*/> rng;
  // CHECK: DPCT1050:{{[0-9]+}}: The template argument of the RNG engine could not be deduced. You need to update this code.
  // CHECK: rng = oneapi::mkl::rng::device::philox4x32x10<dpct_placeholder/*Fix the vec_size manually*/>(clock64(), {1234, static_cast<std::uint64_t>(tid * 8)});
  curandState_t rng;
  curand_init(clock64(), tid, 1234, &rng);

  counter[threadIdx.x] = 0;

  for (int i = 0; i < ITERATIONS; i++) {
    //CHECK: oneapi::mkl::rng::device::gaussian<float> distr_ct2;
    //CHECK-NEXT: oneapi::mkl::rng::device::uniform<float> distr_ct1;
    //CHECK-NEXT: float x = oneapi::mkl::rng::device::generate(distr_ct1, rng);
    //CHECK-NEXT: sycl::float2 y = oneapi::mkl::rng::device::generate(distr_ct2, rng);
    float x = curand_uniform(&rng);
    float2 y = curand_normal2(&rng);
    counter[threadIdx.x] += 1 - int(x * x + y.x * y.x);
  }

  if (threadIdx.x == 0) {
    totals[blockIdx.x] = 0;
    for (int i = 0; i < WARP_SIZE; i++) {
      totals[blockIdx.x] += counter[i];
    }
  }
}


int main(int argc, char **argv) {
  int *dOut;
  picount<<<NBLOCKS, WARP_SIZE>>>(dOut);

  int size = 10;
  //CHECK: DPCT1050:{{[0-9]+}}: The template argument of the RNG engine could not be deduced. You need to update this code.
  //CHECK: oneapi::mkl::rng::device::philox4x32x10<dpct_placeholder/*Fix the vec_size manually*/> *RandomStates;
  curandState *RandomStates;
  //CHECK: DPCT1050:{{[0-9]+}}: The template argument of the RNG engine could not be deduced. You need to update this code.
  //CHECK: RandomStates = (oneapi::mkl::rng::device::philox4x32x10<dpct_placeholder/*Fix the vec_size manually*/> *)sycl::malloc_device(size * sizeof(oneapi::mkl::rng::device::philox4x32x10<dpct_placeholder/*Fix the vec_size manually*/>) * 10, q_ct1);
  cudaMalloc((void**)&RandomStates, size * sizeof(curandState) * 10);
  //CHECK: DPCT1050:{{[0-9]+}}: The template argument of the RNG engine could not be deduced. You need to update this code.
  //CHECK: RandomStates = sycl::malloc_device<oneapi::mkl::rng::device::philox4x32x10<dpct_placeholder/*Fix the vec_size manually*/>>(size, q_ct1);
  cudaMalloc((void**)&RandomStates, size * sizeof(curandState));

  return 0;
}

