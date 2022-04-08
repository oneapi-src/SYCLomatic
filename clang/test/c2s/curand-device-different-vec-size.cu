// RUN: c2s --format-range=none -extra-arg-before=-std=c++14 -out-root %T/curand-device-different-vec-size %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
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

  // CHECK: c2s::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<4>> rng;
  // CHECK: rng = c2s::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<4>>(clock64(), {1234, static_cast<std::uint64_t>(tid * 8)});
  curandState_t rng;
  curand_init(clock64(), tid, 1234, &rng);

  counter[threadIdx.x] = 0;

  for (int i = 0; i < ITERATIONS; i++) {
    //CHECK: float x = rng.generate<oneapi::mkl::rng::device::uniform<float>, 1>();
    //CHECK-NEXT: sycl::float2 y = rng.generate<oneapi::mkl::rng::device::gaussian<float>, 2>();
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
  //CHECK: c2s::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<4>> *RandomStates;
  curandState *RandomStates;
  //CHECK: RandomStates = (c2s::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<4>> *)sycl::malloc_device(size * sizeof(c2s::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<4>>) * 10, q_ct1);
  cudaMalloc((void**)&RandomStates, size * sizeof(curandState) * 10);
  //CHECK: RandomStates = sycl::malloc_device<c2s::rng::device::rng_generator<oneapi::mkl::rng::device::philox4x32x10<4>>>(size, q_ct1);
  cudaMalloc((void**)&RandomStates, size * sizeof(curandState));

  return 0;
}

