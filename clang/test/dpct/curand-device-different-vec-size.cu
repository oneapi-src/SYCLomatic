// RUN: dpct --format-range=none -extra-arg-before=-std=c++14 -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/curand-device-different-vec-size.dp.cpp --match-full-lines %s

#include <cuda.h>
#include <curand_kernel.h>
#include <cstdio>

const int WARP_SIZE = 32;
const int NBLOCKS = 640;
const int ITERATIONS = 1000000;


__global__ void picount(int *totals) {
  __shared__ int counter[WARP_SIZE];
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  //CHECK: mkl::rng::device::philox4x32x10<PlaceHolder/*Fix the vec_size mannually*/> rng;
  //CHECK: rng = mkl::rng::device::philox4x32x10<PlaceHolder/*Fix the vec_size mannually*/>(clock64(), {1234 * PlaceHolder/*Fix the vec_size mannually*/, tid * 8});
  curandState_t rng;
  curand_init(clock64(), tid, 1234, &rng);

  counter[threadIdx.x] = 0;

  for (int i = 0; i < ITERATIONS; i++) {
    //CHECK: mkl::rng::device::uniform<float> distr_ct{{[0-9]+}};
    //CHECK-NEXT: float x = mkl::rng::device::generate(distr_ct{{[0-9]+}}, rng);
    //CHECK-NEXT: sycl::float2 y = mkl::rng::device::generate(distr_ct{{[0-9]+}}, rng);
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

  return 0;
}